import sys
import os
from .UCF_preprocessing import preprocessing
from .OF_utils import optical_flow_prep
from .dmd_preprocessing import dmd_prep
from .mrdmd_preprocessing import mrdmd_prep

cwd = os.getcwd()
data_dir = os.path.join(cwd,'data')
list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
UCF_dir = os.path.join(data_dir, 'UCF-101')
src_dir = os.path.join(data_dir,'frames')
sequence_length = 10
image_shape = (216,216,3)
if __name__== "__main__":
    arg = sys.argv[1]
    if 'mrdmd' in arg.lower():
        sequence_length=16
        dest_dir = os.path.join(data_dir,'MrDMD_frames')
        preprocessing(list_dir,UCF_dir,dest_dir,sequence_length=sequence_length,image_size=image_shape, overwrite=True,normalization=False,mean_subtraction=False, horizontal_flip=False,random_crop=False,consistent=True,continuous_seq=True)
        src_dir = dest_dir
        dest_dir= os.path.join(data_dir,'MrDMD_images')
        mrdmd_prep(src_dir,dest_dir,12,6,overwrite=True)
    elif 'dmd' in arg.lower():
        dest_dir = os.path.join(data_dir,'DMD_images')
        dmd_prep(src_dir,dest_dir,5,6,overwrite=True)
    elif 'of' in arg.lower():
        dest_dir = os.path.join(data_dir,'flow_images')
        optical_flow_prep(src_dir,dest_dir,overwrite=True)
    else:
        dest_dir = os.path.join(data_dir, 'frames')
        preprocessing(list_dir,UCF_dir,dest_dir,sequence_length=sequence_length,image_size=image_shape, overwrite=True,normalization=False,mean_subtraction=False, horizontal_flip=False,random_crop=False,consistent=True,continuous_seq=True)
