from pandas import DataFrame, Series

from utils.constants import *
from utils.parse_data import *
from utils.ts_alignment import *

from typing import Tuple, Any
import pandas as pd
import torch


def get_libsize(subdir: str) -> int:
    assert subdir in SUBDIRS

    return 700 if subdir.startswith("chaotic") else 420


def get_predsize(subdir: str) -> int:
    assert subdir in SUBDIRS

    return 100 if subdir.startswith("chaotic") else 60


def get_data_in_df(subdir: str, is_reduced: bool = True) -> tuple[pd.DataFrame,
                                                                  pd.DataFrame]:
    assert subdir in SUBDIRS

    # parsing data from accelerometer and hyroscope
    dev_filename = f"./signals/{subdir}.json"
    devices_data = parse_devices(dev_filename)

    # parsing data from video
    vid_filename = f"./encoded_video/{subdir}.json"
    video_data = parse_video(vid_filename)

    # alignment of time series
    if is_reduced:
        devices_data, video_data = cut_devices_ts(devices_data, video_data)
    else:
        devices_data, video_data = expand_video_ts(devices_data, video_data)

    return devices_data, video_data


def get_data_in_tensor(subdir: str, is_reduced: bool = True) -> tuple[torch.Tensor,
                                                                      torch.Tensor]:
    devices_data, video_data = get_data_in_df(subdir, is_reduced)

    # The first column of devices_data is `Time`, so we need to get rid of it
    return torch.Tensor(devices_data.values[:, 1:]), torch.Tensor(video_data.values)
