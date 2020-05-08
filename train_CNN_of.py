import os
from train_CNN import fit_model
from utils.UCF_utils import get_data_list
from models.temporal_CNN import temporal_CNN

N_CLASSES=101
if __name__ == '__main__':
    dataset = 'ucf'
    if len(sys.argv) > 0:
        dataset = sys.argv[0]
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    if 'hmdb' in dataset.lower():
        list_dir = os.path.join(data_dir,'hmdb51_test_train_splits')
    else:
        list_dir = os.path.join(data_dir,'ucfTrainTestlist')
    
    weights_dir = os.path.join('models')
    weights_dir = os.path.join(cwd,'models')
    old_weights_dir = os.path.join(weights_dir, 'temporal_cnn_42.h5')
    new_weights_dir = os.path.join(weights_dir, 'temporal_cnn_43.h5')
    video_dir = os.path.join(data_dir, 'OF_data')
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    input_shape = (216, 216, 18)
    model = temporal_CNN(input_shape, N_CLASSES, new_weights_dir, include_top=True)
    fit_model(model, train_data, test_data, new_weights_dir, input_shape, optical_flow=True)

