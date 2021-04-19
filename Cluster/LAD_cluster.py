### Use K-means to calculate the approximated visuals center of unseen classes

import json
import os
import torch
from torch.utils.data import DataLoader
import torchvision.models as model
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import torch.nn.functional as F
import numpy as np
import argparse
import pickle



def KM(features, center_num, typ, split):
    clf = KMeans(n_clusters=center_num, n_init=50, max_iter=100000, init="k-means++")

    print("Start Cluster ...")
    s=clf.fit(features)
    print("Finish Cluster ...")

    obj={}
    obj["VC"]=clf.cluster_centers_.tolist()

    print('Start writing ...')
    json.dump(obj,open("LAD_VC_ResNet_%s_%d.json"%(typ,split),"w"))
    print("Finish writing ...")


def SC(features,opts):
    Spectral = SpectralClustering(n_clusters=opts.center_num, eigen_solver='arpack', affinity="nearest_neighbors")

    print("Start Cluster ...")
    pred_class = Spectral.fit_predict(features)
    print("Finish Cluster ...")

    belong = Spectral.labels_
    sum = {}
    count = {}
    for i, x in enumerate(features):
        label = belong[i]
        if sum.get(label) is None:
            sum[label] = [0.0] * 2048
            count[label] = 0
        for j, y in enumerate(x):
            sum[label][j] += y
        count[label] += 1

    all_cluster_center = []
    for label in sum.keys():

        for i, x in enumerate(sum[label]):
            sum[label][i] /= (count[label] * 1.0)

        all_cluster_center.append(sum[label])

    print("Start writing ...")
    obj = {}
    url = "C_VC_ResNet_%s_%s.json"%(opts.dataset_name,opts.mode)
    obj["VC"] = all_cluster_center
    json.dump(obj, open(url, "w"))
    print("Finish writing ...")

def cluster(typ, split, lad_bin):

    meta_split_info = np.load(lad_bin + 'meta_split_info_v2.npy', allow_pickle=True).item()
    unseen_classes = meta_split_info['splits'][split][typ]['unseen']

    with open(lad_bin + 'splits/split_{:d}/{:s}/r50_features/unseen_all.pkl'.format(split, typ),
              'rb') as infile:
        unseen_all = pickle.load(infile)
    all_features = []
    for item in unseen_all:
        all_features.append(np.array(item[0]))
    KM(all_features, len(unseen_classes), typ, split)

if __name__ == '__main__':

    lad_bin = '/home/peilin/Dataset/LAD/annotations/npys/'
    lad_bin = '/users/pyu12/data/bats/projects/attributes/LAD/'
    valid_range = ['E', 'V', 'F', 'A', 'H']
    split_range = range(5)

    for split in split_range:
        for typ in valid_range:
            cluster(typ, split, lad_bin)





