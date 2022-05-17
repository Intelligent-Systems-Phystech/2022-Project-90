import pandas as pd

from utils.constants import *


def parse_devices(filename: str) -> pd.DataFrame:
    data = pd.read_json(filename, orient="records")

    columns_to_delete = ['version', 'device name', 'recording time',
                         'platform', 'appVersion', 'device id',
                         'sensors', 'sampleRateMs', 'time',
                         'seconds_elapsed']
    data.drop(columns=columns_to_delete, inplace=True)
    data.drop(data.tail(1).index, inplace=True)

    acc_data = data.loc[data.sensor == 'Accelerometer']
    gyr_data = data.loc[data.sensor == 'Gyroscope']

    acc_data = acc_data.drop(columns=['sensor'])
    gyr_data = gyr_data.drop(columns=['sensor'])
    gyr_data.reset_index(drop=True, inplace=True)

    devices_data = pd.concat([acc_data, gyr_data], axis=1, copy=False)
    devices_data.columns = TARGET_COLUMNS
    devices_data.insert(0, "time", range(1, devices_data.shape[0] + 1))

    return devices_data


def parse_video(filename: str) -> pd.DataFrame:
    video_data = pd.read_json(filename, orient="records")

    columns_to_delete = ['image_id', 'category_id', 'score', 'box', 'idx']
    video_data.drop(columns=columns_to_delete, inplace=True)

    video_data = video_data['keypoints'].apply(pd.Series)
    video_data.drop(columns=range(2, KEYPOINTS_CNT * 3, 3), inplace=True)
    video_data.columns = range(video_data.shape[1])
    video_data.drop(columns=EXTRA_VIDEO_COLS, inplace=True)
    video_data.columns = VIDEO_COLUMNS

    return video_data
