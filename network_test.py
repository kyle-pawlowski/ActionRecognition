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
import matplotlib.pyplot as plt

from models.temporal_CNN import temporal_CNN, dmd_CNN, mrdmd_CNN
from utils.UCF_utils import sequence_generator, get_data_list
from utils.dmd_preprocessing import _stack_dmd
from utils.mrdmd_preprocessing import _stack_dmd as _stack_mrdmd
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
    
def pipeline_test(model, test_data, data_type, class_index, num_classes=101, window_size=3):
    correct = 0
    total = 0
    start_time = time.time()
    correct_cats = np.zeros((num_classes,))
    total_cats = np.zeros((num_classes,))
    for example, datay in test_data:
        example = os.path.splitext(example)[0] + '.npy'
        datax = np.load(example)
        if 'mrdmd' in data_type.lower():
            processed = _stack_mrdmd(datax, window_size, -1, deeper=False, condensed=True)
        elif 'hybrid' in data_type.lower():
            processed = _stack_mrdmd(datax, window_size, -1, deeper=False, condensed=True,hybrid=True)
        elif 'dmd' in data_type.lower():
            processed = _stack_dmd(datax,window_size,-1,deeper=False,condensed=True)
        else:
            processed = stack_optical_flow(datax)
        processed = np.reshape(processed, (1,)+processed.shape)
        answer = model.predict(processed)
        maximum = np.max(answer)
        answer = np.equal(answer,maximum)
        if answer[0,datay-1] == 1:
            correct+=1
            correct_cats[datay-1] += 1
        total+=1
        total_cats[datay-1] += 1
    time_taken = time.time()-start_time
    np.where(total_cats==0, 1, total_cats)
    correct_cats = correct_cats / total_cats
    
    #make graph
    plt.style.use('ggplot')
    x = list(class_index.keys())
    x_pos = range(correct_cats.shape[0])
    plt.bar(x_pos, correct_cats, color='pink')
    plt.xlabel('Category')
    plt.ylabel('% Correct')
    plt.title('Accuracy by Class')
    plt.xticks(x_pos, x)
    plt.setp(plt.gca().get_xticklabels(), rotation=90, horizontalalignment='right')
    plt.rc('xtick', labelsize='xx-small')
    
    print('Test Accuracy: ' + str(correct*100/total))
    print('It took ' + str(time_taken/60) + ' minutes to test on ' + str(total) + ' sequences!')
    plt.savefig('test_result.png', bbox_inches="tight")
    plt.show()
        
if __name__ == '__main__':
    datatype = 'dmd'
    window_size = 3
    sequence_length =16
    weights_name = 'dmd_testing_21682.h5'
    dataset='ucf'
    multitask=False
    if len(sys.argv) > 1:
        datatype = sys.argv[1]
        if len(sys.argv) > 2:
            dataset = sys.argv[2]
            if len(sys.argv) > 3:
                  window_size = int(sys.argv[3])
                  if len(sys.argv) > 4:
                      multitask = int(sys.argv[4])
                      if len(sys.argv) >5:
                          weights_name = sys.argv[5]
        
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    if 'hmdb' in dataset.lower():
        list_dir = os.path.join(data_dir, 'hmdb51_test_train_splits')
    else:
        list_dir = os.path.join(data_dir,'ucfTrainTestlist')
    weights_dir = os.path.join(cwd,'models')
    weights_dir = os.path.join(weights_dir, weights_name)
    if 'mrdmd' in datatype.lower():
        video_dir = os.path.join(data_dir,'UCF-DMD-Testing')
        input_shape= (216, 216, sequence_length-window_size+1)
        model = mrdmd_CNN(input_shape, (N_CLASSES, 51), weights_dir, include_top=True, is_training=False, multitask=multitask,for_hmdb=('hmdb' in dataset.lower()))
    elif 'of' in datatype.lower():
        video_dir = os.path.join(data_dir,'UCF-DMD-Testing')
        input_shape = (216,216,2*sequence_length-2)
        model = temporal_CNN(input_shape,(N_CLASSES, 51),weights_dir,include_top=True, is_training=False, multitask=multitask, for_hmdb=('hmdb' in dataset.lower()))
    elif 'hybrid' in datatype.lower():
        video_dir = os.path.join(data_dir,'UCF-DMD-Testing')
        input_shape= (216, 216, sequence_length-window_size+1)
        model = mrdmd_CNN(input_shape, (N_CLASSES, 51), weights_dir, include_top=True, is_training=False, multitask=multitask,for_hmdb=('hmdb' in dataset.lower()), hybrid=True)
    else:
        video_dir = os.path.join(data_dir,'UCF-DMD-Testing')
        input_shape = (216,216,sequence_length-window_size+1)
        model = dmd_CNN(input_shape, (N_CLASSES,51), weights_dir, include_top=True, multitask=multitask,for_hmdb=('hmdb' in dataset.lower()), is_training=False)
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    
    #test_dmd(model,test_data)
    pipeline_test(model,test_data,datatype, class_index, window_size=window_size)
