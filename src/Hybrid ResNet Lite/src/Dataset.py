import pcl
import os
import numpy as np
import copy
import torch

from torch.utils.data import Dataset

from utils import pointcloud_augmentation_all, \
                     pointcloud_augmentation_individual_1,\
                     pointcloud_augmentation_individual_2,\
                     pointcloud_crop, \
                     pointcloud_get_props, \
                     pointcloud_get_gts,\
                     pointcloud_get_input_data


class PointCloudInstance(object):
    """
    This is pcd_file instance.
    """
    def __init__(self, pcd, ann=None,
                 up_bound=None, down_bound=0, lateral_bound=None,
                 test=False):

        self.pcd_path = pcd
        self.ann_path = ann
        self.pcd_id = os.path.basename(pcd)
        self.test = test
        self.up_bound = up_bound
        self.down_bound = down_bound
        self.lateral_bound = lateral_bound

        self.pasre_data()

    def pasre_data(self):
        cloud_raw = pcl.load(self.pcd_path)
        cloud_np = np.asarray(cloud_raw)
        cloud_np = np.delete(cloud_np, -1, axis=1)
        # fileter out zeros points
        dis = np.sqrt(cloud_np[:, 0]**2 + cloud_np[:, 1]**2)
        cloud_np = cloud_np[dis > 0.05]
        # cloud_np = np.load(self.pcd_path)
        if not self.test:
            anns_list = []
            with open(self.ann_path, 'r') as ann_f:
                anns = list(ann_f)
                for line in anns:
                    line = line.split('\t')[1:10]
                    if len(line) == 9:
                        line = list(map(float, line))
                    else:
                        continue
                    line = np.asarray(line)
                    line = line[[1, 2, 4, 5, 8]]
                    line[4] = line[4] if 0 <= line[4] < 180 else line[4] + 180
                    if min(line[0:2]) > 0.6 and max(line[0:2]) < 8 \
                    and (1 <= line[0]*line[1] <= 40):
                        anns_list.append(line)
            if len(anns_list) == 0:
                anns_list = None
                anns_np = None
            else:
                anns_np = np.array(anns_list)
        else:
            anns_np = None
        self.anns_np = anns_np
        self.cloud_np = cloud_np
    
    def get_data(self):
        # get pcd_id
        pcd_id = self.pcd_id
        # get pcd_points, pcd_anns
        if not self.test and False:
            # augmentation
            if np.random.rand() < 0.5:
                if np.random.rand() < 0.5:
                    pcd_points, anns = pointcloud_augmentation_all(self.cloud_np, self.anns_np, 1)
                else:
                    pcd_points, anns = pointcloud_augmentation_all(self.cloud_np, self.anns_np, 0)
            else:
                pcd_points, anns = self.cloud_np, self.anns_np
        else:
            pcd_points, anns = self.cloud_np, self.anns_np
        
        pcd_points, pcd_anns = pointcloud_crop(pcd_points, anns, self.up_bound, self.down_bound, self.lateral_bound)

        # get pcd_props
        pcd_props = pointcloud_get_props(pcd_points)
        if pcd_props is None:
            return pcd_id, pcd_points, None, None, None
        
        if not self.test and np.random.rand() > 0.1:
            pcd_points, pcd_props = pointcloud_augmentation_individual_1(pcd_points, pcd_props)

        # get prop_gts
        pcd_gts = pointcloud_get_gts(pcd_points, pcd_props, pcd_anns)

        if not self.test and np.random.rand() > 0.1:
            pcd_points, pcd_gts = pointcloud_augmentation_individual_2(pcd_points, pcd_props, pcd_gts)
        
        # get input_data, the input_data has translation invariance.
        # and update pcd_props with prop_scales.
        pcd_input_data = pointcloud_get_input_data(pcd_points, pcd_props)

        if pcd_anns is None:
            pcd_anns = []

        pcd_props = pcd_props.reshape(-1, 5)
        # set the positive and negative sample ratio 
        positive_args = np.where(pcd_gts[:, 0] == 1)
        negative_args = np.where(pcd_gts[:, 0] == 0)
        ratio_num = 2
        if ratio_num * len(positive_args) < len(negative_args):
            pos_num = 1 if len(positive_args) == 0 else len(positive_args)
            neg_num = len(negative_args)
            neg_pos = np.random.choice(neg_num, size=ratio_num*pos_num, replace=True if neg_num < ratio_num*pos_num else False)
            neg_sample = negative_args[neg_pos]
            pos_sample = positive_args
            prop_sample = np.vstack((pos_sample, neg_sample))
        else:
            prop_sample = np.argwhere(pcd_gts[:, 0] >=0)

        return pcd_id, pcd_points, pcd_props, pcd_gts, pcd_input_data, pcd_anns

class VehcileDataset(Dataset):
    
    def __init__(self, pcd_dir, ann_dir,
                 multiplier=1,
                 up_bound=None, down_bound=0, lateral_bound=None, 
                 test=False, transform=None):
        self.pcd_dir = pcd_dir
        self.ann_dir = ann_dir
        self.pcd_up_bound = up_bound
        self.pcd_down_bound = down_bound
        self.pcd_lateral_bound = lateral_bound
        self.test = test
        self.multiplier = multiplier
        self.transform = transform

        self.data_parse()
        
        if not test:
            self.parse_reg_norm()

    def data_parse(self):
        pcd_file_list = os.listdir(self.pcd_dir)
        self.data_list = []
        for pcd_f in pcd_file_list:

            pcd_p = os.path.join(self.pcd_dir, pcd_f)
            
            if not self.test:
                ann_f = pcd_f.replace(".pcd", ".txt")
                ann_p = os.path.join(self.ann_dir, ann_f)
            else:
                ann_p = None
            pcd_ins = PointCloudInstance(pcd_p, ann_p, test=self.test, 
                            up_bound=self.pcd_up_bound,
                            down_bound=self.pcd_down_bound, lateral_bound=self.pcd_lateral_bound)
            # pcd_ins = PointCloudInstance(pcd_p, ann_p, test=self.test)
            
            self.data_list.append(pcd_ins)

    def __len__(self):
        return len(self.data_list) * self.multiplier

    def __getitem__(self, idx):

        rel_idx = idx % len(self.data_list)

        pcd_id, pcd_points, pcd_props, pcd_gts, pcd_input_data, pcd_anns = self.data_list[rel_idx].get_data()
        if pcd_props is None:
            return pcd_id, 0

        sample = {'point':torch.from_numpy(pcd_points),
                'input_data':torch.from_numpy(pcd_input_data),
                'ground_truth':torch.from_numpy(pcd_gts),
                'proposal':torch.from_numpy(pcd_props),
                'pcd_anns':pcd_anns}
        if self.transform is not None:
            sample = self.transform(sample)

        return pcd_id, sample

    def parse_reg_norm(self):
        gts_tmp = []
        if not self.test:
            self.test = True
            for i in range(len(self.data_list)):
                pcd_id, pcd_points, pcd_props, pcd_gts, pcd_input_data, pcd_anns = self.data_list[i].get_data()
                if len(pcd_gts) > 0:
                    gts_tmp.append(pcd_gts)
            self.test = False
        else:
            for i in range(len(self.data_list)):
                pcd_id, pcd_points, pcd_props, pcd_gts, pcd_input_data, pcd_anns = self.data_list[i].get_data()
                if len(pcd_gts) > 0:
                    gts_tmp.append(pcd_gts)

        gts_np = np.concatenate(gts_tmp, axis=0).reshape(-1, 13)
        postive_pos = np.argwhere(gts_np[:, 0] >= 1)

        std_gts = np.std(gts_np[postive_pos, 1:5], axis=0)
        men_gts = np.mean(gts_np[postive_pos, 1:5], axis=0)

        self.reg_norm = np.vstack((std_gts, men_gts)).T