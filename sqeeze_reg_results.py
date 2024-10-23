import sys
from pathlib import Path
import shutil
import os
import csv
import numpy as np
import re

initial_angles = np.arange(0, 360, 15)

kept_angles = np.arange(0, 180, 15)
# kept_angles = np.arange(0, 180, 30)
# kept_angles = np.arange(0, 180, 60)
# kept_angles = np.arange(0, 180, 180)

# dataset_name = 'S3_v3_reg_results'
dataset_name = 'S3_test_reg_results'

datapath_initial = Path('/home/d.sorokin/dev/geology/input/' + dataset_name)

datapath_reduced = Path('/home/d.sorokin/dev/geology/input/' + dataset_name + '_' + str(len(kept_angles)))

filename_prefix = 'moved_'

datapath_reduced.mkdir(exist_ok=True)
(datapath_reduced / 'imgs').mkdir(exist_ok=True)

shutil.copytree(str(datapath_initial / 'valid_zones'), str(datapath_reduced / 'valid_zones'), dirs_exist_ok=True)

for img_path in sorted(list((datapath_initial / 'imgs').iterdir())):
    print(f'Processing {img_path}')
    for img_angle_name in sorted(list(img_path.iterdir())):
        m = re.search('moved_(.+?).jpg', str(img_angle_name))
        if (int(m.group(1)) in list(kept_angles)):
            print(f'Copying {str(img_angle_name)}')
            # print('copy {0} to {1}'.format(str(img_angle_name), str(datapath_reduced / 'imgs' / img_path.name / img_angle_name.name)))
            (datapath_reduced / 'imgs' / img_path.name).mkdir(exist_ok=True, parents=True)
            shutil.copy(str(img_angle_name), str(datapath_reduced / 'imgs' / img_path.name / img_angle_name.name))
