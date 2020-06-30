import os
import torch
import copy
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import argparse

from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter

from Dataset import VehcileDataset
from network import VehicleDetection


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

parser = argparse.ArgumentParser("2D LiDAR Based Vehcile Detection Training Script")
parser.add_argument("--init_lr", type=float, default=1*1e-3)#1*1e-3
parser.add_argument("--lr_step", type=int, default=10)#10
parser.add_argument("--train_data_dir", type=str, default="../data/train/")
parser.add_argument("--test_data_dir", type=str, default="../data/test/")
parser.add_argument("--pretrained_model", type=str, default=None)
args = parser.parse_args()

init_lr = args.init_lr
lr_step = args.lr_step


def main():
    # configure dataset and dataloader
    train_pcd_dir = os.path.join(args.train_data_dir, "pcd")
    train_ann_dir = os.path.join(args.train_data_dir, "bbox")

    test_pcd_dir = os.path.join(args.test_data_dir, "pcd")
    test_ann_dir = os.path.join(args.test_data_dir, "bbox")

    # train_whole_set
    train_dataset = VehcileDataset(train_pcd_dir, train_ann_dir,
                        up_bound=40, lateral_bound=16.7, down_bound=-3.33, multiplier=3)

    test_dataset = VehcileDataset(test_pcd_dir, test_ann_dir,
                        up_bound=40, lateral_bound=16.7, down_bound=-3.33, test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=6)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=6)

    dataloaders = {'train':train_loader, 'test':test_loader}
    dataset_sizes = {'train':len(train_dataset), 'test':len(test_dataset)}


    gt_norm_dict = train_dataset.reg_norm


    # configure or load models:

    model = VehicleDetection().double()

    # load the pretraining model:
    if args.pretrained_model is not None:
        
        model.load_state_dict(torch.load(args.pretrained_model))
        model.train()
        print("Successfully load the pretrained model!")

    model = model.to(device)

    criterion_1 = nn.BCELoss()

    criterion_2 = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1*1e-5)

    print("Start Training")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.8)
    model_ft = train_model(model, dataloaders, dataset_sizes, criterion_1, criterion_2,\
                          optimizer, exp_lr_scheduler, gt_norm_dict)
    
def train_model(model, dataloaders, dataset_sizes, criterion_1, criterion_2,\
                optimizer, scheduler, gt_norm_dict, num_epoches=200):
    
    criterion_3 = nn.MSELoss()

    train_log_prefix = 'train_' + time.strftime("%m-%d-%H-%M", time.localtime())


    test_log_prefix = 'test_' + time.strftime("%m-%d-%H-%M", time.localtime())

    tensor_board_dir = train_log_prefix
    tensor_board_dir = os.path.join("./new_train_logs/", tensor_board_dir)
    os.makedirs(tensor_board_dir)
    
    best_model = None
    bast_acc = 0
    best_pr, best_pr_pre, best_pr_rec = .0, .0, .0
    
    batch_size = 1

    tensorboard_writer = SummaryWriter(tensor_board_dir)

    nor_rev_std = torch.Tensor(np.diag(gt_norm_dict[:, 0].T)).double().to(device)
    nor_rev_men = torch.Tensor(gt_norm_dict[:, 1].T).double().to(device)

    loss_loc_param = 1.0
    loss_cls_param = 1.0
    
    min_loss = 1e4

    for epoch in tqdm(range(num_epoches)):
        print('-'*70)
        print('Epoch: {}/{}'.format(epoch + 1, num_epoches))
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            elif phase == 'val' or phase == 'test':
                model.eval()
            else:
                raise ValueError("WRONG DATALOADER PHASE: {}".format(phase))

            running_loss = 0.0
            running_loss_cls = 0.0
            running_loss_loc = 0.0

            sample_num = 0
            dis_thresh= 0.45 
            theta_thersh = np.pi / 9
            
            acc_den, acc_num = 0, 0 
            iter_idx = 0

            for pcd_id, data in tqdm(dataloaders[phase]):
                iter_idx += 1
                inputs = data['input_data'].double().to(device)
                props = data['proposal'].double().to(device)
                gts = data['ground_truth'].double().to(device).view(1, -1, 13)

                if props.size(1) == 0:
                    continue

                optimizer.zero_grad()
                if torch.sum(gts[:, :, 0]).item() == 0:
                    continue

                with torch.set_grad_enabled(phase=='train'):
                    

                    inputs.transpose_(1, 2)

                    loc_pre, cls_pre = model(inputs, props)

                    cls_ = torch.sigmoid(cls_pre)
                    
                    cls_ = torch.clamp(cls_, min=0.0001, max=1.0)
                    cls_ = cls_.view(1, -1).squeeze(0)
                    loc_ = loc_pre.view(-1, 4)
                    loc_ = torch.mm(loc_, nor_rev_std) + nor_rev_men

                    with torch.no_grad():

                        cls_np = cls_.cpu().numpy()
                        if not (np.logical_and(cls_np > np.zeros_like(cls_np), cls_np < np.ones_like(cls_np))).all():
                            print(cls_)
                            continue

                        pre_cls = (cls_ > 0.5)
                        pre_cls = pre_cls.long().to(device)
                        gt_cls = gts[:, :, 0].long().view(1, -1).squeeze(0)


                        theta_ = torch.atan2(loc_[:, 3], loc_[:, 2]).view(1, -1)

                        comp = (pre_cls == gt_cls)
                        all_truth = torch.full_like(gt_cls, 1).to(device)
                        all_false = torch.full_like(gt_cls, 0).to(device)
                        
                        dis_err = (gts[:, :, 1:3] - loc_[:, 0:2]).view(-1, 2).to(device)
                        dis_err = torch.sqrt(dis_err[:, 0]**2 + dis_err[:, 1]**2)

                        dis_positive = (dis_err < dis_thresh)
                        theta_err = (gts[:, :, 9] - theta_).squeeze()
                        
                        the_positive_1 = torch.abs(theta_err) <= theta_thersh
                        the_positive_2 = torch.abs(theta_err + np.pi) <= theta_thersh
                        the_positive_3 = torch.abs(theta_err - np.pi) <= theta_thersh

                        theta_positive = (the_positive_1 | the_positive_2 | the_positive_3)

                        dis_negative = ~dis_positive
                        theta_negative = ~theta_positive

                        cls_positive = (pre_cls == all_truth)
                        cls_negative = ~cls_positive

                        pre_positive = (cls_positive & dis_positive & theta_positive)
                        pre_negative = ~pre_positive

                        gt_positive = (gt_cls == all_truth)

                        acc_den += torch.sum(all_truth).item()
                        acc_num += torch.sum(gt_cls == pre_cls).item()
                                      
                    loss_cls_param = 1.5
                    loss_loc_param = 1.5
                    
                    loc_.unsqueeze_(0)
                    if cls_.size() == torch.Size([]):
                        cls_ = cls_.unsqueeze(0)

                    loss_cls = criterion_1(cls_, gt_cls.double())

                    if torch.sum(gt_positive).item() > 0:

                        loss_loc = criterion_2(loc_[:, gt_positive, 0:4], gts[:, gt_positive, 1:5]) + \
                                   criterion_3(loc_[:, gt_positive, 0:4], gts[:, gt_positive, 1:5])
                    else:
                        loss_loc = 0
                    loss = loss_loc_param * loss_loc + loss_cls_param * loss_cls
                    
                    if torch.isnan(loss):
                        print(loss_cls)
                        print(loss_loc)
                        print(loc_[:, gt_positive, 0:6])
                        print(gts[:, gt_positive, 1:7])

                    if phase == 'train':

                        loss.backward()

                        clip_grad_value_(model.parameters(), 5)
                        
                        optimizer.step()
                    

                running_loss += loss.item()
                running_loss_cls += loss_cls.item()
                running_loss_loc += loss_loc.item()
                sample_num += gts.size(1)


            # print(sample_num)
            epoch_loss = running_loss / dataset_sizes[phase]

            print('')
            print("Binary Classification Acc: {:.4f}".format(acc_num / (acc_den + 1e-3)))
            print("Loss: {:.4f}".format(epoch_loss))

            if phase == 'train':
                tensorboard_writer.add_scalar("{}/runnning_loss".format(train_log_prefix), running_loss / dataset_sizes[phase], epoch)
                tensorboard_writer.add_scalar("{}/loc loss".format(train_log_prefix), running_loss_loc / dataset_sizes[phase], epoch)
                tensorboard_writer.add_scalar("{}/cls loss".format(train_log_prefix), running_loss_cls / dataset_sizes[phase], epoch)
                tensorboard_writer.flush()

            elif phase == 'test':
                tensorboard_writer.add_scalar("{}/runnning_loss".format(test_log_prefix), running_loss / dataset_sizes[phase], epoch)
                tensorboard_writer.add_scalar("{}/loc loss".format(test_log_prefix), running_loss_loc / dataset_sizes[phase], epoch)
                tensorboard_writer.add_scalar("{}/cls loss".format(test_log_prefix), running_loss_cls / dataset_sizes[phase], epoch)
                tensorboard_writer.flush()

            if phase == 'test' and  epoch_loss < min_loss:
                
                min_loss = epoch_loss
                best_ws = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                torch.save(best_ws, './'+ tensor_board_dir + '/' + train_log_prefix +'model_train_best_check.pth')
                print("best_check_point {} saved.".format(epoch))
        
        scheduler.step()

    print('Best epoch num: {}'.format(best_epoch))
    model.load_state_dict(best_ws)
    return model

if __name__ == "__main__":
    main()
