import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of naughty log

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

import config
from metrics import iou_tf
from unet import res_unet, weightedLoss
from utils.base import MaskLoadParams, prepare_experiment
from utils.callbacks import TestCallback, TestCallbackPolarized
from utils.generators import AutoBalancedPatchGenerator, SimpleBatchGenerator, AutoBalancedPatchGeneratorPolarized
from utils.patches import combine_from_patches, split_into_patches

import json

def set_gpu(gpu_index):
	import tensorflow as tf
	physical_devices  = tf.config.experimental.list_physical_devices('GPU')
	print(f'Available GPUs: {len(physical_devices )}')
	if physical_devices:
		print(f'Choosing GPU #{gpu_index}')
		try:
			tf.config.experimental.set_visible_devices([physical_devices[gpu_index]], 'GPU')
			logical_devices = tf.config.list_logical_devices('GPU')
			assert len(logical_devices) == 1
			print(f'Success. Now visible GPUs: {len(logical_devices)}')
		except RuntimeError as e:
			print('Something went wrong!')
			print(e)
                        

def fix_seed():
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    # # The below is necessary for starting Numpy generated random numbers
    # # in a well-defined initial state.
    # np.random.seed(123)

    # # The below is necessary for starting core Python generated random numbers
    # # in a well-defined state.
    # python_random.seed(123)

    # # The below set_seed() will make random number generation
    # # in the TensorFlow backend have a well-defined initial state.
    # # For further details, see:
    # # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    # tf.random.set_seed(1234)


class GeoModel:

    def __init__(self, n_pol, patch_size, batch_size, offset, n_classes, LR, patch_overlay, class_weights=None):
        self.n_pol = n_pol
        self.patch_s = patch_size
        self.batch_s = batch_size
        self.offset = offset
        self.n_classes = n_classes
        self.model = None
        self.LR = LR
        self.patch_overlay = patch_overlay
        self.class_weights = class_weights
        if self.class_weights is None:
            self.class_weights = [1 / self.n_classes] * self.n_classes

    def initialize(self, n_filters, n_layers):
        self.model = res_unet(
            # (None, None, 3),
            (None, None, 3 * (self.n_pol + 1)),
            n_classes=self.n_classes,
            BN=True,
            filters=n_filters,
            n_layers=n_layers,
        )

        # self.model.compile(
        #     optimizer=Adam(learning_rate=self.LR), 
        #     loss = weightedLoss(categorical_crossentropy, self.class_weights),
        #     metrics=[iou_tf]
        # )

    def load(self, model_path):
        # self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.load_weights(model_path)

    def predict_image(self, img: np.ndarray):
        patches = split_into_patches(img, self.patch_s, self.offset, self.patch_overlay) 
        init_patch_len = len(patches)

        while (len(patches) % self.batch_s != 0):
            patches.append(patches[-1])
        pred_patches = []

        for i in range(0, len(patches), self.batch_s):
            batch = np.stack(patches[i : i+self.batch_s])
            prediction = self.model.predict_on_batch(batch)
            for x in prediction:
                pred_patches.append(x)
        
        pred_patches = pred_patches[:init_patch_len]
        result = combine_from_patches(pred_patches, self.patch_s, self.offset, self.patch_overlay, img.shape[:2])
        return result
    

    def train(
        self, train_gen, val_gen, n_steps, epochs, val_steps,
        test_img_folder: Path, test_mask_folder: Path, test_img_folder_polarized: Path, test_mask_folder_polarized: Path, test_output: Path,
        codes_to_lbls, lbls_to_colors, mask_load_p: MaskLoadParams
    ):

        callback_test = TestCallbackPolarized(
            test_img_folder, test_mask_folder, test_img_folder_polarized, test_mask_folder_polarized, 
            lambda img: self.predict_image(img),
            test_output, codes_to_lbls, lbls_to_colors, self.offset, mask_load_p, self.n_pol
        )
        
        (test_output / 'models').mkdir()
        checkpoint_path = str(test_output / 'models' / 'best.hdf5')

        callback_checkpoint = ModelCheckpoint(
            monitor='val_loss', filepath=checkpoint_path, save_best_only=True, save_weights_only=True
        )

        _ = self.model.fit(
            train_gen,
            validation_data=val_gen, 
            steps_per_epoch=n_steps,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=[
                callback_test,
                callback_checkpoint,
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4),
            ],
        )


class TrainConfig(object):
    def __init__(self, n_steps=800, epochs=50, val_steps=80, num_threads=16) -> None:
        self.n_steps = n_steps
        self.epochs = epochs
        self.val_steps = val_steps
        self.num_threads = num_threads

class NetConfig(object):
    def __init__(self, patch_s = 384, batch_s = 16, n_layers = 4, n_filters = 16, LR = 0.001, patch_overlay = 0.5) -> None:
        self.patch_s = patch_s
        self.batch_s = batch_s
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.LR = LR
        self.patch_overlay = patch_overlay

class ExpConfig(object):
    def __init__(self, 
                 dataset_name : str = 'S1_v1_and_S3_v2', 
                 n_polazied : int = 3, 
                 add_dataset_pol_name : str = 'S3_v2_reg_results',
                 enable_add_img : bool = True, 
                 cache_path : str= 'cache/maps/', 
                 exp_out_path : str = 'output_pol',
                 data_path : str = '/home/d.sorokin/dev/geology/input/',
                 train_config : TrainConfig = TrainConfig(),
                 net_config : NetConfig = NetConfig()) -> None:
        self.dataset_name = dataset_name
        self.n_polazied = n_polazied
        self.add_dataset_pol_name = add_dataset_pol_name
        self.enable_add_img = enable_add_img
        self.cache_path = cache_path
        self.exp_out_path = exp_out_path
        self.data_path = data_path
        self.train_config = train_config
        self.net_config = net_config

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
    
    def to_json_file(self, path : Path):
        with open(path, 'w') as outfile:
            outfile.write(self.to_json())
    
    @staticmethod
    def from_json_file(path : Path):
        with open(path) as f:
            j = json.load(f)
        exp_config = ExpConfig(**j)
        exp_config.net_config = NetConfig(**j['net_config'])
        exp_config.train_config = TrainConfig(**j['train_config'])
        return exp_config


def run_experiment(exp_config : ExpConfig):
    
    dataset_name = exp_config.dataset_name # 'S1_v1_and_S3_v2'
    n_polazied = exp_config.n_polazied # 3
    add_dataset_pol_name = exp_config.add_dataset_pol_name # 'S3_v2_reg_results'
    enable_add_img = exp_config.enable_add_img # True
    cache_path = exp_config.cache_path # 'cache/maps/'
    exp_out_path = exp_config.exp_out_path # 'output_pol',
    data_path = exp_config.data_path # '/home/d.sorokin/dev/geology/input/'
    train_config = exp_config.train_config
    net_config = exp_config.net_config

    # patch_s = 384
    # batch_s = 16
    # n_layers = 4
    # n_filters = 16
    # LR = 0.001
    # patch_overlay = 0.5

    # data_path = '/home/d.sorokin/dev/geology/input/'
    # cache_path = 'cache/maps/'
    
    exp_path = prepare_experiment(Path(exp_out_path))
    exp_config.to_json_file(Path(exp_path / 'exp_config.json'))

    # dataset_name = 'S1_v1_and_S3_v2'
    dataset_name_base = dataset_name + '_base'
    dataset_name_pol = dataset_name + '_polarized'

    # n_polazied = 3
    # enable_add_img = True
    add_dataset_pol_name = add_dataset_pol_name + '_' + str(n_polazied)
    assert Path(data_path + add_dataset_pol_name).exists(),  f'folder with additional pol image does not exist: {add_dataset_pol_name}'

    # missed_classes = (3, 5, 7, 9, 10, 12) # for S1_v1
    missed_classes = (5, 7) # for S3_v1, S3_v2, S3_v3

    n_classes = len(config.class_names)
    present_classes = tuple(i for i in range(len(config.class_names)) if i not in missed_classes)
    n_classes_sq = len(present_classes)

    pg = AutoBalancedPatchGeneratorPolarized(
        Path(data_path + 'dataset/' + dataset_name + '/imgs/train/'),
        Path(data_path + 'dataset/' + dataset_name + '/masks/train/'),
        Path(data_path + cache_path),
        Path(data_path + add_dataset_pol_name + '/imgs'),
        Path(data_path + add_dataset_pol_name + '/valid_zones'),
        net_config.patch_s, n_classes=n_classes_sq, enable_add_img=enable_add_img, distancing=0.5, mixin_random_every=5)

    missed_classes = pg.get_missed_classes()
    print('==== missed classes:', missed_classes)

    # pg.benchmark()

    # exit(0)

    squeeze_code_mappings = {code: i for i, code in enumerate(present_classes)}
    codes_to_lbls = {i: config.class_names[code] for i, code in enumerate(present_classes)}

    mask_load_p = MaskLoadParams(None, squeeze=True, squeeze_mappings=squeeze_code_mappings)

    # loss_weights = recalc_loss_weights_2(pg.get_class_weights(remove_missed_classes=True))

    bg = SimpleBatchGenerator(pg, net_config.batch_s, mask_load_p, augment=True)
    # bg.save_test_batch('/home/d.sorokin/dev/geology/tmp/', train_config.num_threads)
    # return

    model = GeoModel(n_polazied, net_config.patch_s, net_config.batch_s, offset=8, n_classes=n_classes_sq, LR=net_config.LR, patch_overlay=net_config.patch_overlay, class_weights=None)
    model.initialize(net_config.n_filters, net_config.n_layers)
    # # model.load(Path('./output/exp_89/models/best.hdf5'))

    model.model.compile(
        optimizer=Adam(learning_rate=net_config.LR),
        # loss = weightedLoss(categorical_crossentropy, loss_weights),
        loss=categorical_crossentropy,
        metrics=[iou_tf]
    )

    model.train(
        # bg.g_balanced(), bg.g_random(), n_steps=400, epochs=50, val_steps=80,
        # bg.g_balanced(), bg.g_random(), n_steps=800, epochs=50, val_steps=80,
        # bg.g_balanced(), bg.g_random(), n_steps=10, epochs=5, val_steps=5,
        bg.g_balanced(train_config.num_threads), bg.g_random(train_config.num_threads), n_steps=train_config.n_steps, epochs=train_config.epochs, val_steps=train_config.val_steps,
        test_img_folder=Path(data_path + '/dataset/' + dataset_name_base + '/imgs/test/'),
        test_mask_folder=Path(data_path + '/dataset/' + dataset_name_base + '/masks/test/'),
        test_img_folder_polarized=Path(data_path + '/dataset/' + dataset_name_pol + '/imgs/test/'),
        test_mask_folder_polarized=Path(data_path + '/dataset/' + dataset_name_pol + '/masks/test/'),
        test_output=exp_path, codes_to_lbls=codes_to_lbls, lbls_to_colors=config.lbls_to_colors,
        mask_load_p=mask_load_p
    )


if __name__ == "__main__":
    assert len(sys.argv) > 1 and sys.argv[1].isnumeric()
    gpu_index = int(sys.argv[1])
    assert gpu_index >= 0
    set_gpu(gpu_index)

    fix_seed()

    assert len(sys.argv) > 2 and Path(sys.argv[2]).exists(), f'config file {sys.argv[2]} does not exist'

    exp_config = ExpConfig.from_json_file(sys.argv[2])
    print('===== running experiment with config: =====')
    print(exp_config.to_json())

    run_experiment(exp_config)
