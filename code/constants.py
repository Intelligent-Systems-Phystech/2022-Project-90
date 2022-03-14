import numpy as np


E = 3
THETA = 2

TARGET_COLUMNS = ['acc_z', 'acc_y', 'acc_x', 'gyr_z', 'gyr_y', 'gyr_x']
TARGET_SIZE = len(TARGET_COLUMNS)

KEYPOINTS_CNT = 68
VIDEO_COLUMNS = np.ravel([[f"x_{i}", f"y_{i}"] for i in range(KEYPOINTS_CNT)])

SUBDIRS = ('round_and_round', 'chaotic_1', 'chaotic_2', 'chaotic_3', 
           'cyclic_1', 'cyclic_2')
PRED_METHODS = ('ccm', 'pls', 'cca', 'naive')
