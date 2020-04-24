from utils.UCF_preprocessing import preprocessing
import os
cwd= os.getcwd()
data_dir = os.path.join(cwd,'data')
list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
UCF_dir = os.path.join(data_dir, 'UCF-101')
dest_dir = os.path.join(data_dir,'UCF-Preprocessed')
sequence_length = 10
image_shape = (216,216,3)
preprocessing(list_dir, UCF_dir, dest_dir, sequence_length, image_shape, overwrite=True,random_crop=False, horizontal_flip=False, continuous_seq=True,mean_subtraction=False)
