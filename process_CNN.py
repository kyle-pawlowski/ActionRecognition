import os
from models.finetuned_resnet import finetuned_resnet
from utils.model_processing import model_processing

N_CLASSES = 101
IMSIZE = (216, 216, 3)


if __name__ == '__main__':
    src_dir = '/home/kyle/Documents/Research/dmd_object_tracking_study/data/action_recognition/UCF-101'
    dest_dir = '/home/kyle/Documents/Research/ActionRecognition/data/CNN_Predicted'
    weights_dir = '/home/kyle/Documents/Research/ActionRecognition/models'

    finetuned_resnet_weights = os.path.join(weights_dir, 'finetuned_resnet.h5')
    model = finetuned_resnet(include_top=False, weights_dir=finetuned_resnet_weights)

    TIMESEQ_LEN = 10
    model_processing(model, src_dir, dest_dir, TIMESEQ_LEN)
