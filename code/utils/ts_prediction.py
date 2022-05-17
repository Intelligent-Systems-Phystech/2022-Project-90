import pyEDM as edm
import pandas as pd
import numpy as np

from utils.constants import *


def naive_predict_ts(data, libsize, predsize, Tp, subdir):
    assert subdir in SUBDIRS
    
    lib = f"1 {libsize}"
    pred = f"{libsize+1} {libsize+predsize}"
    
    for i, col in enumerate(TARGET_COLUMNS):
        edm.SMap(dataFrame=data, 
                lib=lib, 
                pred=pred, 
                predictFile=f"preds/{subdir}/naive/{col}_pred.csv",
                embedded=False, 
                E=E, 
                target=col, 
                columns=[col], 
                theta=THETA, 
                Tp=Tp,
                showPlot=False);

        
def ccm_predict_ts(multi_data, extra_feat_inds, libsize, predsize, Tp, subdir):
    assert subdir in SUBDIRS
    
    lib = f"1 {libsize}"
    pred = f"{libsize+1} {libsize+predsize}"
    
    for i, col in enumerate(TARGET_COLUMNS):
        edm.SMap(dataFrame=multi_data, 
                lib=lib, 
                pred=pred,  
                predictFile=f"preds/{subdir}/ccm/{col}_pred.csv",
                embedded=True, 
                target=col, 
                columns=list(VIDEO_COLUMNS[extra_feat_inds[i]])+[col], 
                theta=THETA, 
                Tp=Tp,
                showPlot=False);


def linear_predict_ts(latent_video, devices_data, libsize, predsize,
                      Tp, subdir, latent_mode="pls"):
    assert subdir in SUBDIRS
    assert latent_mode in ('pls', 'cca')
    
    lib = f"1 {libsize}"
    pred = f"{libsize+1} {libsize+predsize}"
    
    tmp_video_df = pd.DataFrame(latent_video)
    tmp_video_df.columns = [str(num) for num in range(latent_video.shape[1])]
    multi_data = pd.concat([devices_data, tmp_video_df], axis=1, copy=False)
    
    for i, col in enumerate(TARGET_COLUMNS):
        edm.SMap(dataFrame=multi_data, 
                lib=lib, 
                pred=pred,  
                predictFile=f"preds/{subdir}/{latent_mode}/{col}_pred.csv",
                embedded=True, 
                target=col, 
                columns=list(tmp_video_df.columns)+[col], 
                theta=THETA, 
                Tp=Tp,
                showPlot=False);
