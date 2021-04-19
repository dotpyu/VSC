import torch
import numpy as np
import torch.nn.functional as F
import json
import pickle

# lad_bin = '/home/peilin/Dataset/LAD/annotations/npys/'

lad_bin = '/users/pyu12/data/bats/projects/attributes/LAD/splits/'
def extract(typ='E', split=0):
    meta_split_info = np.load(lad_bin + 'split_meta_info.npy', allow_pickle=True).item()
    seen_classes = meta_split_info['seen'][typ]
    unseen_classes = meta_split_info['unseen'][typ]

    # with open(lad_bin + 'seen_all_labels.pkl', 'rb') as infile:
    #     seen_all_labels = pickle.load(infile)

    with open(lad_bin + 'split_{:d}/{:s}/r50_features/seen_all.pkl'.format(split, typ),
              'rb') as infile:
        seen_all = pickle.load(infile)

    with open(lad_bin + 'split_{:d}/{:s}/r50_features/unseen_all.pkl'.format(split, typ),
              'rb') as infile:
        unseen_all = pickle.load(infile)

    seen = {label:[] for label in seen_classes}
    unseen = {label: [] for label in unseen_classes}
    # for idx in range(len(seen_all_labels)):
    #     try:
    #         if seen_all_labels[idx] in seen_classes:
    #             seen[seen_all_labels[idx]].append((seen_a[idx][0][seen_all_labels[idx]]))
    #     except ValueError:
    #         print(idx)
    #         print(seen_a[idx])
    for item in seen_all:
        try:
            label = int(item[1])
            if label in seen_classes:
                seen[label].append(item[0])
        except ValueError:
            print(item)

    for item in unseen_all:
        try:
            label = int(item[1])
            if label in unseen_classes:
                unseen[label].append(item[0])
        except ValueError:
            print(item)

    target_VC = []
    for x in seen_classes:
        features = seen[x]
        sum = [0.0] * 2048
        sum = np.array(sum)
        cnt = 0
        for y in features:
            cnt += 1
            sum += np.array(y)
        sum /= cnt
        avg = torch.tensor(sum)
        avg = F.normalize(avg, dim=0)
        target_VC.append(avg.numpy().tolist())

    test_VC = []
    for x in unseen_classes:
        features = unseen[x]
        sum = [0.0] * 2048
        sum = np.array(sum)
        cnt = 0
        for y in features:
            cnt += 1
            sum += np.array(y)
        sum /= cnt
        avg = torch.tensor(sum)
        avg = F.normalize(avg, dim=0)
        target_VC.append(avg.numpy().tolist())

    obj = {}
    obj["train"] = target_VC
    obj["test"] = test_VC
    cur_url = "LAD_ResNet50_VC_{:s}_{:d}.json".format(typ, split)
    json.dump(obj, open(cur_url, "w"))


if __name__ == '__main__':
    valid_range = ['E', 'V', 'F', 'A', 'H']
    split_range = range(5)

    for split in split_range:
        for typ in valid_range:
            extract(typ, split)