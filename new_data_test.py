# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:53:38 2020

@author: Pawlowski
"""

import os

from PIL import Image

from utils.UCF_utils import process_clip

def vids_to_npy(vids):
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    src_dir = os.path.join(data_dir,'Sport_Test')
    dest_dir = os.path.join(data_dir,'Sport_Test_Preprocessed')
    for vid in vids:
        process_clip(os.path.join(src_dir,vid),os.path.join(dest_dir,vid), 16, (216,216,3),
                     horizontal_flip=False,random_crop=False, continuous_seq=True)
        
if __name__ is '__main__':
    vids = ['sports1.avi', 'sports2.avi','sports3.avi','sports4.avi']
    vids_to_npy(vids)
    
