from utils.UCF_preprocessing import regenerate_data
import os
import sys
dataset = 'ucf'
if len(sys.argv) > 1:
    dataset = sys.argv[1]

cwd = os.getcwd()
data_dir = os.path.join(cwd,'data')
if 'hmdb' in dataset.lower():
    list_dir = os.path.join(data_dir,'hmdb51_test_train_splits')
    UCF_dir = os.path.join(data_dir,'hmdb51_org')
else:
    list_dir = os.path.join(data_dir,'ucfTrainTestlist')
    UCF_dir = os.path.join(data_dir,'UCF-101')
regenerate_data(data_dir,list_dir,UCF_dir,temporal='MrDMD')
