import os

import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create
from util.data_util import data_prepare


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1,labeled_point=0.1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop, self.labeled_point = split, voxel_size, transform, voxel_max, shuffle_index, loop, labeled_point
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            #self.data_list = [item for item in data_list if ('Area_{}'.format(2) in item or 'Area_{}'.format(3) in item or 'Area_{}'.format(4) in item) and ('_1' in item and '_11' not in item)]
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
            #self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item and ('_1' in item and '_11' not in item)]

        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

        self.num_classes=list(range(13))+[255]
    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        
        
        # labeled point
        if self.split=='train' or self.split=='trainval':
            if '%' in self.labeled_point:
                r = float(self.labeled_point[:-1]) / 100
                #print('----------------------')
                num_pts = data.shape[0]
                num_with_anno = max(int(num_pts * r), 1)
                num_without_anno = num_pts - num_with_anno
                idx_without_anno = np.random.choice(num_pts, num_without_anno, replace=False)
                data[idx_without_anno,6]=255 #Unlabeled
                #
            else:
                for i in self.num_classes:
                    ind_per_class = np.where(data[:,6] == i)[0]  # index of points belongs to a specific class
                    num_per_class = len(ind_per_class)
                    if num_per_class > 0:
                        num_with_anno = int(self.labeled_point) #max(int(num_per_class * r), 1) #int(labeled_point)
                        num_without_anno = num_per_class - num_with_anno
                        idx_without_anno = np.random.choice(ind_per_class, num_without_anno, replace=False)
                        data[idx_without_anno,6] = 255

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop
