import os
import numpy as np
import SharedArray as SA

import torch
from torch.utils.data import Dataset

from util.voxelize import voxelize
from util.data_util import sa_create, collate_fn
from util.data_util import data_prepare_scannet
import glob

class Scannetv2(Dataset):
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1,labeled_point=0.1):
        super().__init__()

        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop
        self.labeled_point=labeled_point

        if split == "train" or split == 'val':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        elif split == 'trainval':
            self.data_list = glob.glob(os.path.join(data_root, "train", "*.pth")) + glob.glob(os.path.join(data_root, "val", "*.pth"))
        else:
            raise ValueError("no such split: {}".format(split))
            
        print("voxel_size: ", voxel_size)
        print("Totally {} samples in {} set.".format(len(self.data_list), split))

    def __getitem__(self, idx):
        # data_idx = self.data_idx[idx % len(self.data_idx)]

        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]
        data = torch.load(data_path)

        coord, feat = data[0], data[1]
        label = data[2]
        
        if self.split != 'test':
            label = data[2]
        
        if self.split=='train' or self.split=='trainval':
            #print('----------------------')
            if '%' in self.labeled_point:
                r = float(self.labeled_point[:-1]) / 100
                num_pts = len(data[0]) #data.shape[0]
                num_with_anno = max(int(num_pts * r), 1)
                num_without_anno = num_pts - num_with_anno
                idx_without_anno = np.random.choice(num_pts, num_without_anno, replace=False)
                label[idx_without_anno]= -100 #Unlabeled
            else:
                num_pts = len(data[0]) #data.shape[0]
                num_with_anno = int(self.labeled_point)
                num_without_anno = num_pts - num_with_anno
                idx_without_anno = np.random.choice(num_pts, num_without_anno, replace=False)
                label[idx_without_anno] = -100
        
        coord, feat, label = data_prepare_scannet(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label

    def __len__(self):
        # return len(self.data_idx) * self.loop
        return len(self.data_list) * self.loop