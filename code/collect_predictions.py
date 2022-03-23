import pandas as pd

from constants import *


def collect_pred_from_diff_methods(subdir, target_col):
    preds = None

    for i, pred_method in enumerate(PRED_METHODS):
        usecols = ['Predictions']

        if i == 0:
            usecols = ['Observations', 'Predictions']

        tmp_df = pd.read_csv(f"preds/{subdir}/{pred_method}/{target_col}_pred.csv", 
                             usecols=usecols)

        if i == 0:
            tmp_df['Observations'] = tmp_df.Observations.shift(1)
            preds = tmp_df
        else:
            preds[pred_method] = tmp_df.iloc[:, 0].values
    
    preds.drop(index=[0], inplace=True)
    preds.columns = ['target'] + list(PRED_METHODS)
    
    return preds


def collect_pred_from_diff_targets(subdir, pred_method):
    preds = None

    for i, target in enumerate(TARGET_COLUMNS):
        usecols = ['Observations', 'Predictions']

        tmp_df = pd.read_csv(f"preds/{subdir}/{pred_method}/{target}_pred.csv", 
                             usecols=usecols)
        tmp_df['Observations'] = tmp_df.Observations.shift(1)

        if i == 0:
           preds = tmp_df
        else:
           preds = pd.concat([preds, tmp_df], axis=1)
     
    preds.drop(index=[0], inplace=True)
    preds.columns = np.ravel([(target, f"{target}_pred") for target in TARGET_COLUMNS])
    
    return preds
