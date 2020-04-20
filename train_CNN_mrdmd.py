import os
import keras.callbacks
from utils.UCF_utils import image_from_sequence_generator, sequence_generator, get_data_list
from models.finetuned_resnet import finetuned_resnet
from models.temporal_CNN import dmd_CNN
from keras.optimizers import SGD
from utils.UCF_preprocessing import regenerate_data
import threading
from train_CNN_dmd import fit_model
N_CLASSES = 101
BatchSize = 6 

if __name__ == '__main__':
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    weights_dir = os.path.join('models')


    # fine tune resnet50
    # train_data = os.path.join(list_dir, 'trainlist.txt')
    # test_data = os.path.join(list_dir, 'testlist.txt')
    video_dir = os.path.join(data_dir, 'UCF-Preprocessed-OF')
    #train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    #input_shape = (10, 216, 216, 3)
    #weights_dir = os.path.join(weights_dir, 'finetuned_resnet_RGB_65.h5')
    #model = finetuned_resnet(include_top=True, weights_dir=weights_dir)
    #fit_model(model, train_data, test_data, weights_dir, input_shape)
    
    # train CNN using dmd as input
    dmd_weights_dir = os.path.join(weights_dir, 'mrdmd_cnn_42.h5')
    of_weights_dir = os.path.join(weights_dir, 'temporal_cnn_42.h5')
    video_dir = os.path.join(data_dir, 'MrDMD_data')
    input_shape = (216,216,7)
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    model = dmd_CNN(input_shape, N_CLASSES, dmd_weights_dir, include_top=True)
    fit_model(model, train_data, test_data, dmd_weights_dir, input_shape, optical_flow=False)
