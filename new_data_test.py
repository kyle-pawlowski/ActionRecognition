# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:53:38 2020

@author: Pawlowski
"""

import os
import sys

from PIL import Image
import numpy as np

from utils.UCF_preprocessing import process_clip
from models.temporal_CNN import dmd_CNN, mrdmd_CNN, temporal_CNN
from utils.dmd_preprocessing import _stack_dmd

def vids_to_npy(vids,src_dir,dest_dir):
    for vid in vids:
        process_clip(os.path.join(src_dir,vid)+'.avi',os.path.join(dest_dir,vid)+'.npy', 16, (216,216,3),
                     horizontal_flip=False,random_crop=False, continuous_seq=True)
        
if __name__ is '__main__':
    datatype = sys.argv[1]
    weights_name = sys.argv[2]
    multitasking = int(sys.argv[3])
    
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    src_dir = os.path.join(data_dir,'Sport_Test')
    dest_dir = os.path.join(data_dir,'Sport_Test_Preprocessed')
    list_dir = os.path.join(data_dir,'ucfTrainTestlist')
    weights_dir = os.path.join(cwd,'models')
    weights_dir = os.path.join(weights_dir,weights_name)
    
    vids = ['sports1', 'sports2','sports3','sports4']
    '''#vids = [os.path.join(dest_dir,vid) for vid in vids]
    vids_to_npy(vids,src_dir,dest_dir)'''
    
    vids = [vid+'.npy' for vid in vids]
    vids = [os.path.join(dest_dir,vid) for vid in vids]
    vids = [np.load(vid) for vid in vids]
    processed = [_stack_dmd(vid, 3, -1, deeper=False) for vid in vids]
    processed = np.array(processed)
    
    if 'mrdmd' in datatype.lower():
        model = mrdmd_CNN((216,216,9), 101, weights_dir, is_training=False)
    elif 'of' in datatype.lower():
        model = temporal_CNN((216,216,30), 101, weights_dir, is_trainin=False)
    else:
        model = dmd_CNN((216,216,14), 101, weights_dir, is_training=False, multitasking=multitasking)
    predictions = model.predict(processed, batch_size=4)
    top_cats = np.ndarray((predictions.shape[0], 5))
    for i in range(top_cats.shape[1]):
        maxes = np.max(predictions,axis=1,keepdims=True)
        top = np.equal(predictions,maxes)
        indexes = np.argwhere(top)
        top_cats[:,i] = indexes[:, 1]
        predictions = np.where(top, -1, predictions)
    
    class_index = []
    class_dir = os.path.join(list_dir, 'classInd.txt')
    with open(class_dir) as fo:
        for line in fo:
            class_number, class_name = line.split()
            class_index.append(class_name)
            
    top_cats_str = np.ndarray(top_cats.shape).astype(np.object)
    for i in range(top_cats.shape[0]):
        for j in range(top_cats.shape[1]):
            top_cats_str[i,j] = class_index[int(top_cats[i,j])-1]
    print(top_cats_str)
            
    
    
