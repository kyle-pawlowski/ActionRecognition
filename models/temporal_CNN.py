from keras.layers import Input
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Model
import os


def temporal_CNN(input_shape, classes, weights_dir, include_top=True):
    '''
    The CNN for optical flow input.
    Since optical flow is not a common image, we cannot finetune pre-trained ResNet (The weights trained on imagenet is
    for images and thus is meaningless for optical flow)
    :param input_shape: the shape of optical flow input
    :param classes: number of classes
    :return:
    '''
    optical_flow_input = Input(shape=input_shape)

    x = Convolution2D(96, kernel_size=(7, 7), strides=(2, 2), padding='same', name='tmp_conv1')(optical_flow_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', name='tmp_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)

    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv4')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)

    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv5')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),dim_ordering="tf")(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='tmp_fc6')(x)
    x = Dropout(0.7)(x)

    x = Dense(2048, activation='relu', name='tmp_fc7')(x)
    x = Dropout(0.7)(x)

    if include_top:
        x = Dense(classes, activation='softmax', name='tmp_fc101')(x)

    model = Model(inputs=optical_flow_input, outputs=x, name='temporal_CNN')

    if os.path.exists(weights_dir):
        model.load_weights(weights_dir, by_name=True)

    return model

def dmd_CNN(input_shape, classes, weights_dir, include_top=True, multitask=False, for_hmdb=False):
    '''
    The CNN for optical flow input.
    Since optical flow is not a common image, we cannot finetune pre-trained ResNet (The weights trained on imagenet is
    for images and thus is meaningless for optical flow)
    :param input_shape: the shape of optical flow input
    :param classes: number of classes or tuple of (number of ucf classes, number of hmdb classes)
    :return:
    '''
    if type(classes) == tuple:
        ucf_classes = classes[0]
        hmdb_classes = classes[1]
    else:
        ucf_classes = classes
        
    dmd_input = Input(shape=input_shape)

    x = Convolution2D(96, kernel_size=(7, 7), strides=(2, 2), padding='same', name='tmp_conv1')(dmd_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', name='tmp_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    #x = Dropout(0.7)(x)
    
    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv4')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    #x = Dropout(0.7)(x)

    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv5')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),dim_ordering="tf")(x)

    ucf = Flatten()(x)
    ucf = Dense(4096, activation='relu', name='tmp_fc6')(ucf)
    #ucf = Dropout(0.5)(ucf)
    
    if multitask:
        hmdb = Flatten()(x)
        hmdb = Dense(4096, activation='relu', name='tmp_fc8')(hmdb)
        #hmdb = Dropout(0.5)(hmdb)

    ucf = Dense(2048, activation='relu', name='tmp_fc7')(ucf)
    #ucf = Dropout(0.5)(ucf)
    
    if multitask:
        hmdb = Dense(2048, activation='relu', name='tmp_fc9')(hmdb)
        #hmdb = Dropout(0.5)(hmdb)

    if include_top:
        ucf = Dense(ucf_classes, activation='softmax', name='tmp_fc101')(ucf)
        if multitask:
            hmdb = Dense(hmdb_classes, activation='softmax', name='tmp_fc51')(hmdb)
            
    if not multitask or not for_hmdb:
        model = Model(inputs=dmd_input, outputs=ucf, name='temporal_CNN')
    else:
        model = Model(inputs=dmd_input, outputs=hmdb,name='temporal_CNN')

    if os.path.exists(weights_dir):
        model.load_weights(weights_dir, by_name=True)

    return model

def mrdmd_CNN(input_shape, classes, weights_dir, include_top=True):
    '''
    The CNN for optical flow input.
    Since optical flow is not a common image, we cannot finetune pre-trained ResNet (The weights trained on imagenet is
    for images and thus is meaningless for optical flow)
    :param input_shape: the shape of optical flow input
    :param classes: number of classes
    :return:
    '''
    optical_flow_input = Input(shape=input_shape)

    x = Convolution2D(96, kernel_size=(7, 7), strides=(2, 2), padding='same', name='tmp_conv1')(optical_flow_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', name='tmp_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv4')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv5')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),dim_ordering="tf")(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='tmp_fc6')(x)
    #x = Dropout(0.2)(x)

    x = Dense(2048, activation='relu', name='tmp_fc7')(x)
    #x = Dropout(0.2)(x)

    if include_top:
        x = Dense(classes, activation='softmax', name='tmp_fc101')(x)

    model = Model(inputs=optical_flow_input, outputs=x, name='temporal_CNN')

    if os.path.exists(weights_dir):
        model.load_weights(weights_dir, by_name=True)

    return model

if __name__ == '__main__':
    input_shape = (216, 216, 18)
    N_CLASSES = 101
    model = temporal_CNN(input_shape, N_CLASSES, weights_dir='')
    print(model.summary())
