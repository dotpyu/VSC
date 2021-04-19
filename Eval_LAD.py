### Test the model performance

import os
import torch
import torchvision.models as model
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linspace
from matplotlib import cm
import json
import matplotlib.pyplot as plt
import argparse

from PIL import Image
import pickle

def L2_dis(x,y):

    return np.sum((x-y)*(x-y))

def NN_search(x,center):

    ret=""
    MINI=-1
    for c in center.keys():
        tmp=L2_dis(x,center[c])
        if MINI==-1:
            MINI=tmp
            ret=c
        if tmp<MINI:
            MINI=tmp
            ret=c
    return ret


def get_center(typ, split):
    center={}
    file="Pred_Center_{:s}_{:d}.txt".format(typ, split)
    with open(file,"r") as f:
        for i,lines in enumerate(f):
            line=lines.strip().split()
            pp=[float(x) for x in line]
            center[i]=np.array(pp)
    return center



def eval(typ, split, lad_bin):
    meta_split_info = np.load(lad_bin + 'meta_split_info_v2.npy', allow_pickle=True).item()
    unseen_classes = meta_split_info['splits'][split][typ]['unseen']
    seen_classes = meta_split_info['splits'][split][typ]['seen']

    with open(lad_bin + 'splits/split_{:d}/{:s}/r50_features/unseen_all.pkl'.format(split, typ),
              'rb') as infile:
        unseen_all = pickle.load(infile)

    translator = {old_label:new_label for new_label, old_label in enumerate(unseen_classes)}
    center = get_center(typ, split)
    correct = 0
    for fea_vec_pair in unseen_all:  #### Test the image features of each class
        fea_vec=np.array(fea_vec_pair[0])
        label = translator[int(fea_vec_pair[1])]
        ans=NN_search(fea_vec,center)  # Find the nearest neighbour in the feature space
        if ans==label:
            correct+=1
    #assert test_class_num==len(target_class), "Maybe there is someting wrong?"
    acc =  correct/len(unseen_all)
    print("For {:s} and Split {:d}, The final MCA result is {:.5f}".format(typ, split,acc))
    return acc


if __name__=="__main__":
    valid_range = ['E', 'V', 'F', 'A', 'H']
    split_range = range(5)
    lad_bin = '/users/pyu12/data/bats/projects/attributes/LAD/'
    results = {}
    for split in split_range:
        for typ in valid_range:
            results[typ+str(split)] = eval(typ, split, lad_bin)
    np.save('results', results)
