import os
from models.finetuned_resnet import finetuned_resnet
from utils.model_processing import model_processing

N_CLASSES = 101
IMSIZE = (216, 216, 3)


if __name__ == '__main__':
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    src_dir = os.path.join(data_dir,'UCF-Preprocessed')
    dest_dir = os.path.join(data_dir,'CNN_Predicted')
    weights_dir = os.path.join(cwd,'models')

    finetuned_resnet_weights = os.path.join(weights_dir, 'finetuned_resnet.h5')
    model = finetuned_resnet(include_top=False, weights_dir=finetuned_resnet_weights)

    TIMESEQ_LEN = 10
    model_processing(model, src_dir, dest_dir, TIMESEQ_LEN)
