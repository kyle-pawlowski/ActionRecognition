# Action Recognition

This project aims to accurately recognize user's action in a series of video frames through combination of [convolution neural nets](https://en.wikipedia.org/wiki/Convolutional_neural_network), and [long-short term memory neural nets](https://en.wikipedia.org/wiki/Long_short-term_memory).

## Project Overview

- This project explores prominent action recognition models with [UCF-101](http://crcv.ucf.edu/data/UCF101.php) dataset
    
- Perfomance of different models are compared and analysis of experiment results are provided

## Environment Set-up

These are instructions for how to get your computer ready to run this code.

1. Download [Anaconda](https://anaconda.com/distribution). This is a Python package management and virtual environment software. Similar softwares such as PIP (Package Installer for Python) and virtualenv will work too.

2. Create a new environment with the required libraries. The libraries included with Anaconda which are required for this code are numpy, pandas, py-opencv, scipy, matplotlib, tensorflow, keras, pillow, sphinx.
```
conda create --name dmd_env python=3.7 numpy, pandas, py-opencv, scipy, matplotlib, tensorflow, keras, pillow, sphinx
```

3. To start using this environment, it must be activated.
```
source activate dmd_env
```

4. Now the PyDMD library must be installed separately using pip
```
python3 -m pip install pydmd
```

5. Install the UCF-101 dataset by clicking [here](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar). Once the download is finished, move the "UCF-101" folder to the data folder. Watch out because this is 6GB worth of videos. It shouldn't be added to the repository, so the entire data folder is included in the gitignore.

6. Install the labels for the dataset by clicking [here](https://www.crcvucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip). Once the download is finished, move the "ucfTrainTestlist" folder to the data directory.

7. Rename one of the testlist files in the "ucfTrainTestlist" folder to be "testlist.txt" without a number. Rename one of the trainlist files to be "trainlist.txt.

8. Now the files in this repository can be run using `python3 scriptname.py`. Just make sure to run everything from the main directory since that is where the file paths assume the current working directory is. To run a script in a folder, use the package name: `python -m packagename.scriptname`.

## File Structure of the Repo

rnn\_practice: 
    Practices on RNN models and LSTMs with online tutorials and other useful resources

data:
    Training and testing data. (**NOTE: please don't add large data files to this repo, add them to .gitignore**)

models:
    Defining the architecture of models

utils:
    Utils scripts for dataset preparation, input pre-processing and other helper functions 
    
train_CNN:
    Training CNN models. The program loads corresponding models, sets the training parameters and initializes network training

process_CNN:
    Processing video with CNN models. The CNN component is pre-trained and fixed during the training phase of LSTM cells. We can utilize the CNN model to pre-process frames of each video and store the intermediate results for feeding into LSTMs later. This procedure improves the training efficiency of the LRCN model significantly
    
    

train_RNN:
    Training the LRCN model
   
predict:
    Calculating the overall testing accuracy on the entire testing set
    
 

## Models Description

- Fine-tuned ResNet50 and trained solely with single-frame image data. Each frame of the video is considered as an image for training and testing, which generates a natural data augmentation.
   The ResNet50 is from [keras repo](https://github.com/fchollet/deep-learning-models), with weights 
   pre-trained on Imagenet. **./models/finetuned_resnet.py** 
   
![](https://github.com/woodfrog/ActionRecognition/blob/master/readme_imgs/finetuned_resnet.png?raw=true)   

- LRCN (CNN feature extractor, here we use the fine-tuned ResNet50 and LSTMs). The input of LRCN is a sequence of frames uniformly extracted from each video. The fine-tuned ResNet directly uses the result of [1] without extra training (C.F.[Long-term recurrent
   convolutional network](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf)).
   
   **Produce intermediate data using ./process_CNN.py and then train and predict with ./models/RNN.py**
    
![](https://github.com/woodfrog/ActionRecognition/blob/master/readme_imgs/LRCN.png?raw=true)
   
   
- Simple CNN model trained with stacked optical flow data (generate one stacked optical flow from each of the video, and use the optical flow as the input of the network). **./models/temporal_CNN.py**
   
![](https://github.com/woodfrog/ActionRecognition/blob/master/readme_imgs/CNN_optical_flow.png?raw=true)

- Two-stream model, combines the models in [2] and [3] with an extra fusion layer that
   output the final result. [3] and [4] refer to [this paper](http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf)
   **./models/two_stream.py**

![](https://github.com/woodfrog/ActionRecognition/blob/master/readme_imgs/two_stream_model.png?raw=true)

## Citations
If you use this code or ideas from the paper for your research, please cite the following papers:
```
@inproceedings{lrcn2014,
   Author = {Jeff Donahue and Lisa Anne Hendricks and Sergio Guadarrama
             and Marcus Rohrbach and Subhashini Venugopalan and Kate Saenko
             and Trevor Darrell},
   Title = {Long-term Recurrent Convolutional Networks
            for Visual Recognition and Description},
   Year  = {2015},
   Booktitle = {CVPR}
}
@article{DBLP:journals/corr/SimonyanZ14,
  author    = {Karen Simonyan and
               Andrew Zisserman},
  title     = {Two-Stream Convolutional Networks for Action Recognition in Videos},
  journal   = {CoRR},
  volume    = {abs/1406.2199},
  year      = {2014},
  url       = {http://arxiv.org/abs/1406.2199},
  archivePrefix = {arXiv},
  eprint    = {1406.2199},
  timestamp = {Mon, 13 Aug 2018 16:47:39 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/SimonyanZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
