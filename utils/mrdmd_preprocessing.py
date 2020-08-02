import numpy as np
from pydmd import MrDMD
import os
from collections import OrderedDict
import shutil
import warnings
import cv2

from math import log2, floor

def mrdmd_prep(src_dir, dest_dir, window, num_modes, overwrite=False, hybrid=False):
    train_dir = os.path.join(src_dir, 'train')
    test_dir = os.path.join(src_dir, 'test')

    # create dest directory
    if os.path.exists(dest_dir):
        if overwrite:
            shutil.rmtree(dest_dir)
        else:
            raise IOError(dest_dir + ' already exists')
    os.mkdir(dest_dir)
    print(dest_dir, 'created')

    # create directory for training data
    dest_train_dir = os.path.join(dest_dir, 'train')
    if os.path.exists(dest_train_dir):
        print(dest_train_dir, 'already exists')
    else:
        os.mkdir(dest_train_dir)
        print(dest_train_dir, 'created')

    # create directory for testing data
    dest_test_dir = os.path.join(dest_dir, 'test')
    if os.path.exists(dest_test_dir):
        print(dest_test_dir, 'already exists')
    else:
        os.mkdir(dest_test_dir)
        print(dest_test_dir, 'created')

    dir_mapping = OrderedDict(
        [(train_dir, dest_train_dir), (test_dir, dest_test_dir)])  # the mapping between source and dest

    print('Start computing dmds ...')
    for dir, dest_dir in dir_mapping.items():
        print('Processing data in {}'.format(dir))
        for index, class_name in enumerate(os.listdir(dir)):  # run through every class of video
            class_dir = os.path.join(dir, class_name)
            dest_class_dir = os.path.join(dest_dir, class_name)
            if not os.path.exists(dest_class_dir):
                os.mkdir(dest_class_dir)
                # print(dest_class_dir, 'created')
            for filename in os.listdir(class_dir):  # process videos one by one
                file_dir = os.path.join(class_dir, filename)
                frames = np.load(file_dir)
                # note: store the final processed data with type of float16 to save storage
                processed_data = _stack_dmd(frames, window, num_modes,hybrid=hybrid).astype(np.float16)
                dest_file_dir = os.path.join(dest_class_dir, filename)
                np.save(dest_file_dir, processed_data)
            # print('No.{} class {} finished, data saved in {}'.format(index, class_name, dest_class_dir))
 

def _stack_dmd(frames, window, num_modes, grey=True, deeper=False, condensed=True, hybrid=False):
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
        warnings.warn('Warning! The data type has been changed to np.float32 for graylevel conversion...')
    frame_shape = frames.shape[1:-1]  # e.g. frames.shape is (10, 216, 216, 3)i
    num_sequences = frames.shape[0]
    height = frames.shape[1]
    width = frames.shape[2]
    color_ch = frames.shape[3]
    modes = None
    for i in range(num_sequences - window+1):
        selection = frames[i:i+window]
        if grey:
            selection = np.array([cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) for frame in selection])
        mode = _compute_dmd(selection.astype(np.complex128), hybrid=hybrid)
        if modes is None or mode.shape[1]<num_modes: 
            num_modes = mode.shape[1]
            if condensed:
                output_shape = (height,width,num_sequences-window+1)
            elif grey and not deeper:
                output_shape = (height,width*num_modes,num_sequences-window+1)
            elif not grey and deeper:
                output_shape = (num_sequences-window+1,height*color_ch,width,num_modes)  # dmd_modes shape is (139,968, num_modes, num_windows)
            elif grey and deeper:
                output_shape = (num_sequences-window+1,height,width,num_modes)
            else:
                output_shape = (height*color_ch,width*num_modes,num_sequences-window+1)
                
            if modes is None:
                modes = np.ndarray(shape=output_shape)
            elif not condensed:
                if deeper:
                    modes = modes[:,:,:,0:num_modes]
                else:
                    modes = modes[:,0:width*num_modes,:]
        if condensed:
            mode = np.max(mode, axis=1)
            mode = np.reshape(mode,output_shape[:-1])
            modes[:,:,i] = mode
        elif not deeper:
            mode = np.reshape(mode.T[0:num_modes,:],output_shape[0:-1])
            modes[:,:,i]
        else:
            mode = np.reshape(mode.T[0:num_modes,:],output_shape[1:])
            modes[i] = mode
    return modes

def _compute_dmd(frames, hybrid=False):
    window = frames.shape[0]
    if len(frames.shape) == 4:
        vec_frames = np.reshape(frames, (window, frames.shape[1]*frames.shape[2]*frames.shape[3]))
    else:
        vec_frames = np.reshape(frames, (window, frames.shape[1]*frames.shape[2]))
    max_level = floor(log2(window))
    dmd = MrDMD(svd_rank=0, max_level=max_level, max_cycles=1)
    try:
        dmd.fit(vec_frames.T)
    except np.linalg.LinAlgError:
        return np.zeros((frames.shape[1]*frames.shape[2],1))
    if hybrid:
        modes = get_partial_modes(dmd,[0,max_level-1])
        if modes is None:
            return np.zeros((frames.shape[1]*frames.shape[2],1))
    else:
        modes = dmd.modes.real
    return modes

def get_partial_modes(dmd, levels):
    modes = None
    for level in levels:
        try:
            mode = dmd.partial_modes(level).real
        except IndexError:
            print('Invalid Level encountered')
            continue
        print('mode shape: '+str(mode.shape))
        mode = np.max(mode,axis=1,keepdims=True)
        if level > 0 and np.max(abs(mode)) > 0:
            print('Got a non-zero long-term mode')
        if modes is None:
            modes = mode
        else:
            modes = np.maximum(modes,mode)
    return modes
            
 
if __name__ == '__main__':
    sequence_length = 16
    image_size = (216,216,3)
    cwd = os.getcwd()
    src_dir = os.path.join(cwd,'data/UCF-Preprocessed-MrDMD')
    dest_dir = os.path.join(cwd,'data/MrDMD_data')
    mrdmd_prep(src_dir, dest_dir, 8, -1, overwrite=True,hybrid=True) 
