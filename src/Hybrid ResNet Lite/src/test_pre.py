import os
import torch
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader

from Dataset import VehcileDataset
from network import VehicleDetection
from tqdm import tqdm 

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser("2D LiDAR Based Vehcile Detection Training Script")
parser.add_argument("--train_data_dir", type=str, default="../data/train/")
parser.add_argument("--test_data_dir", type=str, default="../data/test/")
parser.add_argument("--pretrained_model", type=str, default=None)
args = parser.parse_args()

def main():

    # Load pretrained model
    pretrain_model_path = args.pretrained_model
    if pretrain_model_path is None:
        print("Please specify the pretrained model path")
        return

    model = VehicleDetection().double()
    pretrain_dicts = torch.load(pretrain_model_path)
    model.load_state_dict(pretrain_dicts)
    model.eval()
    model.to(device)

    # Prepare test dataset.
    test_pcd_dir = os.path.join(args.test_data_dir, "pcd")
    test_pre_dir = os.path.join(args.test_data_dir, "pred")
    if not os.path.isdir(test_pre_dir):
        os.makedirs(test_pre_dir)

    test_dataset = VehcileDataset(test_pcd_dir, None,
                                 up_bound=40, down_bound=-3.33, lateral_bound=16.67,
                                test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # get normalization data
    train_pcd_dir = os.path.join(args.train_data_dir, "pcd")
    train_ann_dir = os.path.join(args.train_data_dir, "bbox")

    train_dataset = VehcileDataset(train_pcd_dir, train_ann_dir,
                            up_bound=40, lateral_bound=16.7, down_bound=-3.33,)
    gt_norm_dict = train_dataset.reg_norm
    nor_rev_std = torch.Tensor(np.diag(gt_norm_dict[:, 0].T)).double().to(device)
    nor_rev_men = torch.Tensor(gt_norm_dict[:, 1].T).double().to(device)

    # inference
    with torch.no_grad():
        for pcd_id, data in tqdm(test_dataloader):
            input_data = data['input_data'].double().to(device)
            props = data['proposal'].double().to(device)
            
            pre_file_path = pcd_id[0][:-4] + '.txt'

            input_data.transpose_(1, 2)
            with open(os.path.join(test_pre_dir, pre_file_path), 'w') as f:

                if props.size(1) == 0 or (input_data.size(-1) <= 60):
                    pass
                else:
                    loc_pre, cls_pre = model(input_data, props)
                    cls_pre = torch.sigmoid(cls_pre)
                    cls_pre = torch.clamp(cls_pre, min=0.001, max=1.0).squeeze().cpu().numpy()

                    loc_pre = loc_pre.view(-1, 4)
                    loc_pre = torch.mm(loc_pre, nor_rev_std) + nor_rev_men
                    
                    theta = torch.atan2(loc_pre[:, 3], loc_pre[:, 2]).squeeze()
                    theta = theta.cpu().numpy()

                    props = props.view(-1, 5)
                    # main_theta = props[:, 5].squeeze().cpu().numpy()
  
                    loc_x = (loc_pre[:, 0] + props[:, 3]).squeeze().cpu().numpy()
                    loc_y = (loc_pre[:, 1] + props[:, 4]).squeeze().cpu().numpy()

                    props_num = props.size(0)
                    for p in range(props_num):
                        if props_num > 1:
                            # if cls_pre[p] > 0.5:
                            pre_ = "{:.4f} \t {:.2f} \t {:.2f} \t{:.2f}\n".format(
                                cls_pre[p], loc_x[p], loc_y[p], theta[p])
                            f.write(pre_)
                        else:
                            # if cls_pre > 0.5:
                            pre_ = "{:.4f} \t {:.2f} \t {:.2f} \t{:.2f}\n".format(
                                cls_pre, loc_x, loc_y, theta)
                            f.write(pre_)

                    
if __name__ == "__main__":
    main()
