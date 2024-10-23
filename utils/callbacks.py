from pathlib import Path
from utils.base import MaskLoadParams
from .eval import TestEvaluator, Tester, TesterPolarized

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


class TestCallback(Callback):
    def __init__(self, img_folder: Path, mask_folder: Path, predict_func, output_path: Path,
                 codes_to_lbls, lbls_to_colors, offset, mask_load_p: MaskLoadParams):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.predict_func = predict_func
        self.lrs = []
        self.evaluator = TestEvaluator(codes_to_lbls, offset)
        self.tester = Tester(self.evaluator, output_path, codes_to_lbls, lbls_to_colors, mask_load_p)

    def on_epoch_end(self, epoch, logs=None):
        description = f'epoch {epoch + 1}'
        self.tester.test_on_set(
            self.img_folder, self.mask_folder, self.predict_func, description)
        self.lrs.append(K.eval(self.model.optimizer.lr))
        self.tester.plot_LR(self.lrs)

class TestCallbackPolarized(Callback):
    def __init__(self, img_folder: Path, mask_folder: Path, img_folder_polarized: Path, mask_folder_polarized: Path, predict_func, output_path: Path,
                 codes_to_lbls, lbls_to_colors, offset, mask_load_p: MaskLoadParams, n_pol):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.img_folder_polarized = img_folder_polarized
        self.mask_folder_polarized = mask_folder_polarized
        self.predict_func = predict_func
        self.lrs = []
        self.evaluator = TestEvaluator(codes_to_lbls, offset)
        self.evaluator_polarized = TestEvaluator(codes_to_lbls, offset)
        self.tester = TesterPolarized(self.evaluator, self.evaluator_polarized, output_path, codes_to_lbls, lbls_to_colors, mask_load_p, n_pol)

    def on_epoch_end(self, epoch, logs=None):
        description = f'epoch {epoch + 1}'
        self.tester.test_on_set(
            self.img_folder, self.mask_folder, self.img_folder_polarized, self.mask_folder_polarized, self.predict_func, description)
        self.lrs.append(K.eval(self.model.optimizer.lr))
        self.tester.plot_LR(self.lrs)
