# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:59:29 2020

@author: Pawlowski
"""
import sys
import os

import keras

from models.temporal_CNN import dmd_CNN
from utils.UCF_utils import sequence_generator, get_data_list

N_CLASSES = 101

def test_dmd(model, test_data):
    BatchSize = 6
    data = sequence_generator(test_data, BatchSize, input_shape, N_CLASSES)
    for x,y in data:
        predictions = model.predict(x)
        
if __name__ == '__main__':
    datatype = 'dmd'
    window_size = 3
    sequence_length =16
    weights_name = 'dmd_cnn_window.h5'
    if len(sys.argv) > 1:
        datatype = sys.argv[1]
        if len(sys.argv) > 2:
              window_size = sys.argv[2]
              if len(sys.argv) >3:
                  weights_name = sys.argv[3]
        
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    list_dir = os.path.join(data_dir,'ucfTrainTestlist')
    weights_dir = os.path.join(cwd,'models')
    weights_dir = os.path.join(weights_dir, weights_name)
    if datatype.lower() is 'mrdmd':
        video_dir = os.path.join(data_dir,'MrDMD_data')
        (216, 216, sequence_length-window_size+1)
    elif datatype.lower() is 'of':
        video_dir = os.path.join(data_dir,'OF_data')
        (216,216,sequence_length+2)
    else:
        video_dir = os.path.join(data_dir,'DMD_data')
        input_shape = (216,216,sequence_length-window_size+1)
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    
    model = dmd_CNN(input_shape, (N_CLASSES,51), weights_dir, include_top=True, multitask=False,for_hmdb=False)
    test_dmd(model,test_data)