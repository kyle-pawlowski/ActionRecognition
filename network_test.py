# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:59:29 2020

@author: Pawlowski
"""
import sys
import os
import time

import keras
import numpy as np

from models.temporal_CNN import temporal_CNN, dmd_CNN
from utils.UCF_utils import sequence_generator, get_data_list
from utils.dmd_preprocessing import _stack_dmd
from utils.OF_utils import stack_optical_flow
N_CLASSES = 101
    
def test_dmd(model, test_data):
    BatchSize = 6
    total=0
    num_correct=0
    data = sequence_generator(test_data, BatchSize, input_shape, N_CLASSES)
    for x,y in data:
        predictions = model.predict(x)
        maxes = np.max(predictions, axis=1, keepdims=True)
        guesses = np.equal(predictions,maxes)
        correct = np.logical_and(guesses,y)
        num_correct += np.sum(correct)
        total += BatchSize
    print('accuracy: ' + str(num_correct*100/total))
    
def pipeline_test(model, test_data, data_type):
    correct = 0
    total = 0
    start_time = time.time()
    for example, datay in test_data:
        example = os.path.splitext(example)[0] + '.npy'
        datax = np.load(example)
        if 'dmd' in data_type.lower():
            processed = _stack_dmd(datax,5,-1,deeper=False,condensed=True)
        else:
            processed = stack_optical_flow(datax)
        processed = np.reshape(processed, (1,)+processed.shape)
        answer = model.predict(processed)
        if answer[0,datay] == 1:
            correct+=1
        total+=1
    time_taken = time.time()-start_time
    print('Test Accuracy: ' + str(correct*100/total))
    print('It took ' + str(time_taken/60) + ' minutes to test on ' + str(total) + ' sequences!')
        
if __name__ == '__main__':
    datatype = 'dmd'
    window_size = 5
    sequence_length =10
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
    if 'mrdmd' in datatype.lower():
        video_dir = os.path.join(data_dir,'MrDMD_data')
        (216, 216, sequence_length-window_size+1)
    elif 'of' in datatype.lower():
        video_dir = os.path.join(data_dir,'UCF-Preprocessed-OF')
        input_shape = (216,216,2*sequence_length-2)
        model = temporal_CNN(input_shape,N_CLASSES,weights_dir,include_top=True, is_training=False)
    else:
        video_dir = os.path.join(data_dir,'UCF-Preprocessed-DMD')
        input_shape = (216,216,sequence_length-window_size+1)
        model = dmd_CNN(input_shape, (N_CLASSES,51), weights_dir, include_top=True, multitask=False,for_hmdb=False, is_training=False)
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    
    #test_dmd(model,test_data)
    pipeline_test(model,test_data,'OF')