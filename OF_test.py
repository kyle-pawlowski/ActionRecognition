from utils.UCF_utils import sequence_generator, get_data_list
import os
import numpy as np
cwd = os.getcwd()
data_dir = os.path.join(cwd,'data')
list_dir = os.path.join(data_dir,'ucfTrainTestlist')
video_dir = os.path.join(data_dir,'UCF-Preprocessed-OF')
input_shape = (10,216, 216,3)
N_CLASSES = 101
train_data, test_data, class_index = get_data_list(list_dir, video_dir)
c = sequence_generator(train_data,32,input_shape,N_CLASSES).__next__()[0]
for i in c:
    print("i shape = " + str(i.shape))
    print(i)
