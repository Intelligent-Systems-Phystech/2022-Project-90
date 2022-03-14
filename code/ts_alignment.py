import numpy as np
import pandas as pd

from constants import *


def cut_devices_ts(devices_data, video_data):
    devices_ts_size = devices_data.shape[0]
    video_ts_size = video_data.shape[0]

    new_devices_ids = np.linspace(0, devices_ts_size, video_ts_size, dtype=int)
    new_devices_ids[-1] -= 1
    cut_devices_data = devices_data.iloc[new_devices_ids]
    
    cut_devices_data.reset_index(drop=True, inplace=True)
    cut_devices_data['time'] = np.arange(1, video_ts_size+1)
    
    return cut_devices_data, video_data


def expand_video_ts(devices_data, video_data):
    devices_ts_size = devices_data.shape[0]
    video_ts_size = video_data.shape[0]

    expansion_coef = math.ceil(devices_ts_size / video_ts_size)
    excess_repeat_cnt = expansion_coef * video_ts_size - devices_ts_size
    indices_to_delete = np.linspace(0, expansion_coef*video_ts_size, excess_repeat_cnt, 
                                    dtype=int)
    indices_to_delete[-1] = expansion_coef*video_ts_size - 1

    expanded_video_data = pd.DataFrame(np.repeat(video_data.values, 
                                                 expansion_coef, 
                                                 axis=0), 
                                       columns=video_data.columns)
    expanded_video_data.drop(index=indices_to_delete, inplace=True)
    expanded_video_data.reset_index(drop=True, inplace=True)
    
    return devices_data, expanded_video_data

