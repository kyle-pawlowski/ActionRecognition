import os
from train_CNN import fit_model
from utils.UCF_utils import get_data_list
from models.temporal_CNN import temporal_CNN

N_CLASSES=101
if __name__ == '__main__':
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'data')
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    weights_dir = os.path.join(cwd,'models')
    weights_dir = os.path.join(weights_dir, 'temporal_cnn_42.h5')
    video_dir = os.path.join(data_dir, 'OF_data')
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    input_shape = (216, 216, 18)
    model = temporal_CNN(input_shape, N_CLASSES, weights_dir, include_top=True)
    fit_model(model, train_data, test_data, weights_dir, input_shape, optical_flow=True)

