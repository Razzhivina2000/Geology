import sys
from pathlib import Path
import shutil
import os
import csv


datapath_raw = Path('/home/d.sorokin/data/geology/raw/Box3-5_DS4/')

datapath_lumenstone = Path('/home/d.sorokin/data/geology/LumenStone/S3_v3')

filename_mapping_path = 'S3_v3_filename_mapping.csv'
imgs_path_src = 'img'
imgs_path_dst = 'imgs'
masks_path_src = 'masks_machine'
masks_path_dst = 'masks'
train_path = 'train'
test_path = 'test'

(datapath_lumenstone).mkdir(exist_ok=True)
(datapath_lumenstone / imgs_path_dst).mkdir(exist_ok=True)
(datapath_lumenstone / imgs_path_dst / train_path).mkdir(exist_ok=True)
(datapath_lumenstone / imgs_path_dst / test_path).mkdir(exist_ok=True)
(datapath_lumenstone / masks_path_dst).mkdir(exist_ok=True)
(datapath_lumenstone / masks_path_dst / train_path).mkdir(exist_ok=True)
(datapath_lumenstone / masks_path_dst / test_path).mkdir(exist_ok=True)


with open(filename_mapping_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        line_count = line_count + 1
        print(row)
        raw_fname = row[0]
        train_idx = row[2]
        test_idx = row[3]
        if train_idx != "" and test_idx == "":
            train_idx = str(int(train_idx)).zfill(2)
            # print('copy {0} to {1}'.format(str(datapath_raw / imgs_path_src / raw_fname), \
            #                                str(datapath_lumenstone / imgs_path_dst / train_path / (train_idx + '.jpg'))))
            # print('copy {0} to {1}'.format(str(datapath_raw / masks_path_src / raw_fname), \
            #                                str(datapath_lumenstone / masks_path_dst / train_path / (train_idx + '.png'))))
            shutil.copy(str(datapath_raw / imgs_path_src / raw_fname), \
                        str(datapath_lumenstone / imgs_path_dst / train_path / (train_idx + '.jpg')))
            shutil.copy(str((datapath_raw / masks_path_src / raw_fname).with_suffix('.png')), \
                        str(datapath_lumenstone / masks_path_dst / train_path / (train_idx + '.png')))
        elif train_idx == "" and test_idx != "":
            test_idx = str(int(test_idx)).zfill(2)
            # print('copy {0} to {1}'.format(str(datapath_raw / imgs_path_src / raw_fname), \
            #                                str(datapath_lumenstone / imgs_path_dst / test_path / (test_idx + '.jpg'))))
            # print('copy {0} to {1}'.format(str(datapath_raw / masks_path_src / raw_fname), \
            #                                str(datapath_lumenstone / masks_path_dst / test_path / (test_idx + '.png'))))
            shutil.copy(str(datapath_raw / imgs_path_src / raw_fname), \
                        str(datapath_lumenstone / imgs_path_dst / test_path / (test_idx + '.jpg')))
            shutil.copy(str((datapath_raw / masks_path_src / raw_fname).with_suffix('.png')), \
                        str(datapath_lumenstone / masks_path_dst / test_path / (test_idx + '.png')))
        else:
            raise('error! both indexes are empty!')

    print(f'Processed {line_count} lines.')


