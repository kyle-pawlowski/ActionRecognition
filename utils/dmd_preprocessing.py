import numpy as np
from pydmd import DMD
import os
from UCF_preprocessing import 

def dmd_prep():
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

    print('Start computing optical flows ...')
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
                processed_data = compute_dmd(frames, mean_sub).astype(np.float16)
                dest_file_dir = os.path.join(dest_class_dir, filename)
                np.save(dest_file_dir, processed_data)
            # print('No.{} class {} finished, data saved in {}'.format(index, class_name, dest_class_dir))
 

def stack_dmd(frames, window, num_modes):
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
        warnings.warn('Warning! The data type has been changed to np.float32 for graylevel conversion...')
    frame_shape = frames.shape[1:-1]  # e.g. frames.shape is (10, 216, 216, 3)
    num_sequences = frames.shape[0]
    output_shape = frame_shape + (2 * (num_sequences - 1),)  # stacked_optical_flow.shape is (216, 216, 18)
    modes = np.ndarray(shape=output_shape)

    for i in range(num_sequences - window):
        
        flow = _calc_optical_flow(prev_gray, next_gray)
        flows[:, :, 2 * i:2 * i + 2] = flow

def compute_dmd(frames, num_modes):
    vec_frames = np.reshape(frames, (frames.shape[0], frames.shape[1]*frames.shape[2]*frames.shape[3]))
    dmd = DMD(svd_rank=num_modes)
    dmd.fit(vec_frames.T)
    modes = np.reshape(dmd.modes.T.real,(frames[1],frames[2],frames[3]))
 
if __name__ == '__main__':
    sequence_length = 10 
    image_size = (216,216,3)
    data_dir = '/home/kyle/Documents/Research/ActionRecognition/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    UCF_dir = os.path.join(data_dir, 'UCF-101')
    frames_dir = os.path.join(data_dir, 'frames/mean.npy')
    
