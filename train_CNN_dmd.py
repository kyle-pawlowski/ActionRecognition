import os
import sys
import keras.callbacks
from utils.UCF_utils import image_from_sequence_generator, sequence_generator, get_data_list
from models.finetuned_resnet import finetuned_resnet
from models.temporal_CNN import dmd_CNN
from keras.optimizers import SGD
from utils.UCF_preprocessing import regenerate_data
import threading

N_CLASSES = 101
BatchSize = 6 

class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()
    def __iter__(self): return self
    def __next__(self):
        self.lock.acquire()
        try:
            return self.it.__next__()
        finally:
            self.lock.release()

def fit_model(model, train_data, test_data, weights_dir, input_shape, optical_flow=False):
    try:
        # using sequence or image_from_sequnece generator
        if optical_flow:
            train_generator = LockedIterator(sequence_generator(train_data, BatchSize, input_shape, N_CLASSES))
            test_generator = LockedIterator(sequence_generator(test_data, BatchSize, input_shape, N_CLASSES))
        else:
            train_generator = LockedIterator(image_from_sequence_generator(train_data, BatchSize, (6,)+input_shape, N_CLASSES))
            test_generator = LockedIterator(image_from_sequence_generator(test_data, BatchSize, (6,)+input_shape, N_CLASSES))

        # frames_dir = '/home/changan/ActionRecognition/data/flow_images'
        # train_generator = image_generator(train_data, frames_dir, BatchSize, input_shape, N_CLASSES, mean_sub=False,
        #                                   normalization=True, random_crop=True, horizontal_flip=True)
        # test_generator = image_generator(test_data, frames_dir, BatchSize, input_shape, N_CLASSES, mean_sub=False,
        #                                  normalization=True, random_crop=True, horizontal_flip=True)

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        print('Start fitting model')
        for i in range(5):
            checkpointer = keras.callbacks.ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
            earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=2, mode='auto')
            tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/try', histogram_freq=0, write_graph=True, write_images=True)
            model.fit_generator(
                train_generator,
                steps_per_epoch=200,
                epochs=1000,
                validation_data=test_generator,
                validation_steps=100,
                verbose=2,
                callbacks=[checkpointer,earlystopping]
            )
            cwd = os.getcwd()
            data_dir = os.path.join(cwd,'data')
            list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
            UCF_dir = os.path.join(data_dir, 'UCF-101')
            regenerate_data(data_dir, list_dir, UCF_dir,temporal='DMD',random=True)

    except KeyboardInterrupt:
        print('Training is interrupted')


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
    
    weights_dir = os.path.join(cwd,'models')


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
    dmd_weights_dir = os.path.join(weights_dir, 'dmd_cnn_reshaped.h5')
    video_dir = os.path.join(data_dir, 'DMD_data')
    input_shape = (216,864,6)
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    model = dmd_CNN(input_shape, N_CLASSES, dmd_weights_dir, include_top=True)
    fit_model(model, train_data, test_data, dmd_weights_dir, input_shape, optical_flow=True)
