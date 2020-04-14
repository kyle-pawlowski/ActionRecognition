from utils.UCF_utils import sequence_generator, get_data_list
import os
import numpy as np
from train_CNN import LockedIterator
cwd = os.getcwd()
data_dir = os.path.join(cwd,'data')
list_dir = os.path.join(data_dir,'ucfTrainTestlist')
video_dir = os.path.join(data_dir,'OF_data')
input_shape = (216, 216,18)
N_CLASSES = 101
train_data, test_data, class_index = get_data_list(list_dir, video_dir)
c = LockedIterator(sequence_generator(train_data,32,input_shape,N_CLASSES))
i = next(c)[0]
print("i shape = " + str(i.shape))
print(i)
