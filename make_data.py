from utils.UCF_preprocessing import regenerate_data
import os

cwd = os.getcwd()
data_dir = os.path.join(cwd,'data')
list_dir = os.path.join(data_dir,'ucfTrainTestlist')
UCF_dir = os.path.join(data_dir,'UCF-101')
regenerate_data(data_dir,list_dir,UCF_dir)
