import pandas as pd
# import numpy as np

from constants import *


def parse_devices(filename):
    data = pd.read_json(filename, orient="records")
    
    COLUMNS_TO_DELETE = ['version', 'device name', 'recording time', 
                     'platform', 'appVersion', 'device id', 
                     'sensors', 'sampleRateMs', 'time', 
                     'seconds_elapsed']
    data.drop(columns=COLUMNS_TO_DELETE, inplace=True)
    data.drop(data.tail(1).index, inplace=True)
    
    acc_data = data.loc[data.sensor == 'Accelerometer']
    gyr_data = data.loc[data.sensor == 'Gyroscope']

    acc_data.drop(columns=['sensor'], inplace=True)
    gyr_data.drop(columns=['sensor'], inplace=True)
    gyr_data.reset_index(drop=True, inplace=True)
    
    devices_data = pd.concat([acc_data, gyr_data], axis=1, copy=False)
    devices_data.columns = TARGET_COLUMNS
    devices_data.insert(0, "time", range(1, devices_data.shape[0]+1))
    
    return devices_data


def parse_video(filename, keypoints_cnt):
    video_data = pd.read_json(filename, orient="records")

    COLUMNS_TO_DELETE = ['image_id', 'category_id', 'score', 'box', 'idx']
    video_data.drop(columns=COLUMNS_TO_DELETE, inplace=True)

    video_data = video_data['keypoints'].apply(pd.Series)
    video_data.drop(columns=range(2, KEYPOINTS_CNT*3, 3), inplace=True)
    video_data.columns = VIDEO_COLUMNS
    
    return video_data
