# coding=utf-8

import json
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment
import time
from Models.GCN import GCN
from torch.optim import lr_scheduler
from Tools.Wasserstein import SinkhornDistance
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random

def l2_loss(a, b):
    return ((a - b)**2).sum() / (len(a) * 2)

def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])


#####  L2+Chamfer-Distance

def CDVSc(a,b,device,n,m,lamda):


    mask=list(range(n-m))
    L2_loss=((a[mask] - b[mask]) ** 2).sum() / ((n-m) * 2) ## L2_Loss of seen classes

    #### Start Calculating CD Loss

    CD_loss=None

    A=a[n-m:]
    B=b[n-m:]

    A=A.cpu()
    B=B.cpu()

    for x in A:
        for y in B:
            dis=((x-y)**2).sum()


    for x in A:
        MINI=None
        for y in B:
            dis=((x-y)**2).sum()
            if MINI is None:
                MINI=dis
            else:
                MINI=min(MINI,dis)
        if CD_loss is None:
            CD_loss=MINI
        else:
            CD_loss+=MINI

    for x in B:
        MINI=None
        for y in A:
            dis=((x-y)**2).sum()
            if MINI is None:
                MINI=dis
            else:
                MINI=min(MINI,dis)
        if CD_loss is None:
            CD_loss=MINI
        else:
            CD_loss+=MINI


    CD_loss=CD_loss.to(device)
    #######

    lamda=0.0003

    tot_loss=L2_loss+CD_loss*lamda
    return tot_loss

#####

def BMVSc(a,b,device,n,m,lamda):

    mask=list(range(n-m))
    L2_loss=((a[mask] - b[mask]) ** 2).sum() / ((n-m) * 2) ## L2_Loss of seen classes


    A=a[n-m:]
    B=b[n-m:]

    DIS=torch.zeros((m,m))


    DIS=DIS.to(device)

    for A_id,x in enumerate(A):
        for B_id,y in enumerate(B):
            dis=((x-y)**2).sum()
            DIS[A_id,B_id]=dis

    matching_loss=0

    cost=DIS.cpu().detach().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    for i,x in enumerate(row_ind):
        matching_loss+=DIS[row_ind[i],col_ind[i]]

    tot_loss=L2_loss+matching_loss*lamda

    return tot_loss


def WDVSc(a,b,device,n,m,lamda,no_use_VSC=True):


    WD=SinkhornDistance(0.01,1000,None,"mean")

    mask=list(range(n-m))
    L2_loss=((a[mask] - b[mask]) ** 2).sum() / ((n-m) * 2) ## L2_Loss of seen classes

    A = a[n - m:]
    B = b[n - m:]

    A=A.cpu()
    B=B.cpu()
    if no_use_VSC:
        WD_loss=0.
        P=None
        C=None
    else:
        WD_loss,P,C=WD(A,B)
        WD_loss = WD_loss.to(device)


    tot_loss=L2_loss+WD_loss*lamda

    return tot_loss,P,C




def get_train_center(url):

    obj=json.load(open(url,"r"))
    VC=obj["train"]
    return VC


def get_cluster_center(url):

    obj=json.load(open(url,"r"))
    test_center=obj["VC"]
    return test_center




def get_attributes(device,att_url,class_url,train_class,test_class):

    attributes=[]
    with open(att_url,"r") as f:
        for lines in f:
            line=lines.strip().split()
            cur=[]
            for x in line:
                y=float(x)
                y=y/100.0
                if y<0.0:
                    y=0.0
                cur.append(y)
            attributes.append(cur)
    ys={}
    pos=0
    with open(class_url,"r") as f:
        for lines in f:
            line=lines.strip().split()
            ys[line[1]]=attributes[pos]
            pos+=1


    ret=[]
    with open(train_class,"r") as f:
        for lines in f:
            line = lines.strip().split()
            ret.append(ys[line[0]])

    with open(test_class,"r") as f:
        for lines in f:
            line = lines.strip().split()
            ret.append(ys[line[0]])


    ret=torch.tensor(ret)
    ret=ret.to(device)

    return ret


def translate_fid2clusters_to_lid2fid(fid2clusters, num_classes):

    lid2fid = {}
    feature_len = len(fid2clusters)
    lid2fid = np.zeros((num_classes, len(fid2clusters)))

    # for lid in range(1, 1+num_classes):
    #     lid2fid[lid] = [0] * feature_len

    # If out of bounds exception, problem with lid > range(num_classes)!
    for fid, clusters in fid2clusters.items():
        for cluster_id, cluster in enumerate(clusters):
            for lid in cluster:
                lid2fid[lid-1][fid] = cluster_id

    return lid2fid


def main_proc(typ, split, lad_bin, save_loc='./'):

    ### Fix the random seed
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    device=torch.device("cuda:0")

    # TODO: Rewrite get_attributes
    # att = get_attributes(device, attributes_url, all_class_url, train_class_url, test_class_url)
    fid2clusters = np.load('/users/pyu12/data/pyu12/datasets/lad/fid2clusters.npy', allow_pickle=True).item()
    sig = translate_fid2clusters_to_lid2fid(fid2clusters, 230)

    meta_split_info = np.load(lad_bin + 'meta_split_info_v2.npy', allow_pickle=True).item()
    unseen_classes = meta_split_info['splits'][split][typ]['unseen']
    seen_classes = meta_split_info['splits'][split][typ]['seen']

    n = len(seen_classes) + len(unseen_classes)  # num_classes
    m = len(unseen_classes)  # num_test_classes
    # with open(lad_bin + 'seen_all_labels.pkl', 'rb') as infile:
    #     seen_all_labels = pickle.load(infile)

    index = {'A':np.array(range(123)), 'F': np.array(range(123, 181)), 'V': np.array(range(181,261)), 'E':np.array(range(261, 337)), 'H':np.array(range(337, 359))}
    train_sig = sig[np.array(seen_classes)-1,:]
    test_sig = sig[np.array(unseen_classes)-1,:]
    att = np.vstack((train_sig, test_sig))
    att = att[:, index[typ]]
    input_dim = len(index[typ])  # num_attributes

    word_vectors = torch.FloatTensor(att).to(device)
    word_vectors = F.normalize(word_vectors)  ## Normalize

    train_center = 'LAD_ResNet50_VC_{:s}_{:d}.json'.format(typ, split)
    cluster_center = 'LAD_VC_ResNet_{:s}_{:d}.json'.format(typ, split)
    VC = get_train_center(train_center)  ## Firstly, to get the necessary training class center
    C_VC = get_cluster_center(cluster_center)  ## Obtain the approximated VC of unseen class
    for x in C_VC:
        VC.append(x)

    VC = torch.tensor(VC)
    VC = VC.to(device)
    VC = F.normalize(VC)


    print('word vectors:', word_vectors.shape)
    print('VC vectors:', VC.shape)

    #####Parameters
    method = 'WDVSc'
    lr = 0.0001
    wd = 0.0005
    max_epoch = 6000
    lamb=0.0003
    hidden_layers = '2048,2048'
    output_dim = 2048
    ####
    edges = []
    edges = edges + [(u, u) for u in range(n)]  ## Set the diagonal to 1

    Net = GCN(n, edges, input_dim, output_dim, hidden_layers, device).to(device)

    optimizer = torch.optim.Adam(Net.parameters(), lr=lr, weight_decay=wd)
    step_optim_scheduler = lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.1)

    # pos=0
    for epoch in range(max_epoch + 1):

        s = time.time()

        Net.train()
        step_optim_scheduler.step(epoch)

        syn_vc = Net(word_vectors)
        if method == 'VCL':
            loss, _, _ = WDVSc(syn_vc, VC, device, n, m, lamb)  ## Here we have set [--no_use_VSC] to True
        if method == 'CDVSc':
            loss = CDVSc(syn_vc, VC, device, n, m, lamb)
        if method == 'BMVSc':
            loss = BMVSc(syn_vc, VC, device, n, m, lamb)
        if method == 'WDVSc':
            loss, _, _ = WDVSc(syn_vc, VC, device, n, m, lamb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        e = time.time()
        print("Epoch %d Loss is %.5f Cost Time %.3f mins" % (epoch, loss.item(), (e - s) / 60))
    #### Training
    Net.eval()
    output_vectors = Net(word_vectors)
    output_vectors = output_vectors.detach()
    file = "Pred_Center_{:s}_{:d}.txt".format(typ, split)
    # pos+=1
    cur = os.getcwd()
    file = os.path.join(save_loc, file)
    with open(file, "w") as f:
        for i in range(m):
            x = i + n - m
            tmp = output_vectors[x].cpu()
            tmp = tmp.numpy()
            ret = ""
            for y in tmp:
                ret += str(y)
                ret += " "
            f.write(ret)
            f.write('\n')


if __name__=='__main__':
    valid_range = ['E', 'V', 'F', 'A', 'H']
    split_range = range(5)
    lad_bin = '/users/pyu12/data/bats/projects/attributes/LAD/'
    for split in split_range:
        for typ in valid_range:
            main_proc(typ, split, lad_bin)