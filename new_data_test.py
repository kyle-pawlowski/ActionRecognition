# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:53:38 2020

@author: Pawlowski
"""

import os

from PIL import Image
import numpy as np

from utils.UCF_preprocessing import process_clip
from models.temporal_CNN import dmd_CNN

def vids_to_npy(vids,src_dir,dest_dir):
    for vid in vids:
        process_clip(os.path.join(src_dir,vid)+'.avi',os.path.join(dest_dir,vid)+'.npy', 16, (216,216,3),
                     horizontal_flip=False,random_crop=False, continuous_seq=True)
        
if __name__ is '__main__':
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    src_dir = os.path.join(data_dir,'Sport_Test')
    dest_dir = os.path.join(data_dir,'Sport_Test_Preprocessed')
    
    '''vids = ['sports1', 'sports2','sports3','sports4']
    #vids = [os.path.join(dest_dir,vid) for vid in vids]
    vids_to_npy(vids,src_dir,dest_dir)'''
    
    vids = ['sports1.npy', 'sports2.npy','sports3.npy','sports4.npy']
    vids = [os.path.join(dest_dir,vid) for vid in vids]
    vids = [np.load(vid) for vid in vids]
    vids = np.array(vids)
    model = dmd_CNN((216,216,14), 101, 'models/dmd_testing_.h5', is_training=False)
    predictions = model.predict(vids, batch_size=4)
    
    
