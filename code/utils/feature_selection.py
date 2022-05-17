import numpy as np
import pandas as pd
import pyEDM as edm
from sklearn.cross_decomposition import PLSCanonical, CCA
from tqdm import tqdm

from utils.constants import *


def read_corr_mat(filename):
    corr_mat = np.array([])

    with open(filename, "r") as file_obj:
        for line in file_obj.readlines():
            line = [float(num) for num in line.split(' ')]
            corr_mat = np.append(corr_mat, line)

    corr_mat = corr_mat.reshape(TARGET_SIZE, -1)
    return corr_mat
    

def ccm_feat_selection(libsize, k_feat_to_select, devices_data, video_data, 
                       read_result=False, save_result=False, filename=""):
    assert (not read_result or not save_result)
    
    if read_result:
        corr_mat = read_corr_mat(filename)
        return corr_mat.argsort(axis=1)[:, -k_feat_to_select:]
    
    libsize = str(libsize)
    corr_mat = np.zeros((video_data.shape[1], devices_data.shape[1]-1))
    multi_data = pd.concat([devices_data, video_data], axis=1, copy=False)

    for i, col in enumerate(tqdm(video_data.columns)):
        for j, target in enumerate(devices_data.columns[1:]):
            # output: LibSize, col:target, target:col
            output = edm.CCM(dataFrame=multi_data, E=E, columns=col, target=target,
                             libSizes=libsize, random=False)
            corr_mat[i, j] = output.values[0, 2]

    corr_mat = corr_mat.T
    extra_feat_inds = corr_mat.argsort(axis=1)[:, -k_feat_to_select:]
    
    if save_result:
        np.savetxt(filename, corr_mat)
    
    return extra_feat_inds


def linear_feat_selection(algo, libsize, devices_data, video_data):
    algo.fit(video_data.iloc[:libsize, :], devices_data.iloc[:libsize, 1:])
    
    return algo.transform(video_data)


def pls_feat_selection(libsize, k_feat_to_select, devices_data, video_data):
    pls_algo = PLSCanonical(n_components=k_feat_to_select)
    
    return linear_feat_selection(pls_algo, libsize, devices_data, video_data)  
    

def cca_feat_selection(libsize, k_feat_to_select, devices_data, video_data):
    cca_algo = CCA(n_components=k_feat_to_select)
    
    return linear_feat_selection(cca_algo, libsize, devices_data, video_data)
