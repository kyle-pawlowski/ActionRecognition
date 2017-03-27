import os
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD
from models.resnet50 import ResNet50

N_CLASSES = 101
IMSIZE = (216, 216, 3)

def finetuned_resnet(include_top, weights_dir):
    '''

    :param include_top: True for training, False for predicting
    :param weights_dir: path to load finetune_resnet.h5
    :return:
    '''
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=IMSIZE)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    if include_top:
        x = Dense(N_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    if os.path.exists(weights_dir):
        model.load_weights(weights_dir, by_name=True)

    return model