# -*- coding:UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

drop_out_prob = 0.4 # 之前是0.4

class Block_Stage_1(nn.Module):
    def __init__(self):
        super(Block_Stage_1, self).__init__()
        self.drop_out = nn.Dropout(p=drop_out_prob)
        self.block = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1),
                                    nn.LeakyReLU(negative_slope=0.05),
                                    # nn.BatchNorm1d(24, True),
                                    nn.Conv1d(64, 64, 1, padding=0),
                                    nn.LeakyReLU(negative_slope=0.05),
                                    # nn.BatchNorm1d(24, True),
                                    nn.Conv1d(64, 64, 3, padding=1))
    def forward(self, input_data):
        output = self.drop_out(self.block(input_data))
        return output

class Block_Stage_2(nn.Module):
    def __init__(self):
        super(Block_Stage_2, self).__init__()
        self.drop_out = nn.Dropout(p=drop_out_prob)
        self.block = nn.Sequential(nn.Conv1d(128, 128, 3, padding=1),
                                    nn.LeakyReLU(negative_slope=0.05),
                                    # nn.BatchNorm1d(48, True),
                                    nn.Conv1d(128, 128, 1, padding=0),
                                    nn.LeakyReLU(negative_slope=0.05),
                                    # nn.BatchNorm1d(48, True),
                                    nn.Conv1d(128, 128, 3, padding=1),
                                    nn.LeakyReLU(negative_slope=0.05))

    def forward(self, input_data):
        output = self.drop_out(self.block(input_data))  
        return output
    
class VehicleDetection(nn.Module):

    def __init__(self):
        super(VehicleDetection, self).__init__()
        self.batchnorm_0 = nn.BatchNorm1d(6)

        self.batchnorm_1 = nn.BatchNorm1d(64)
        self.batchnorm_2 = nn.BatchNorm1d(64)
        self.batchnorm_3 = nn.BatchNorm1d(128)
        self.batchnorm_4 = nn.BatchNorm1d(128)
        self.conv1 = nn.Conv1d(6, 24, padding=0, kernel_size=1)
        self.conv2 = nn.Conv1d(24, 64, padding=0, kernel_size=1)

        self.block_1 = Block_Stage_1()
        self.block_2 = Block_Stage_1()
        
        self.block_cross = nn.Conv1d(64, 128, kernel_size=1, padding=0)
        self.global_pool = nn.AvgPool1d(2)
        
        self.block_3 = Block_Stage_2()
        self.block_4 = Block_Stage_2()

        self.conv_5 = nn.Conv1d(128, 128, kernel_size=1, stride=1, padding=0)

        self.roi_pool = nn.AdaptiveAvgPool1d(5)


        self.sigmoid = nn.Sigmoid()
        self.leaky_relu =  nn.LeakyReLU(negative_slope=0.1)

        self.header_conv1 = nn.Conv1d(128, 128, 3, padding=1)
        self.header_conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.header_conv3 = nn.Conv1d(128, 128, 3, padding=1)

        self.loc_fc_1 = nn.Linear(128*5, 128, bias=True)

        self.loc_fc_2 = nn.Linear(128, 32, bias=True)
        self.loc_fc_3 = nn.Linear(32, 4, bias=True)


        self.cls_fc_1 = nn.Linear(128*5, 128, bias=True)
        self.cls_fc_2 = nn.Linear(128, 32, bias=True)
        self.cls_fc_3 = nn.Linear(32, 1, bias=True)


    def forward(self, input_data, proposals):
        
        data_conv0 = self.conv2(self.conv1(input_data))
        data_bolck1 = self.block_1(data_conv0) + data_conv0
        data_bolck2 = self.block_2(data_bolck1) + data_bolck1

        data_conv3 = self.global_pool(self.block_cross(data_bolck2))
        
        data_block3 = self.block_3(data_conv3) + data_conv3
        data_block4 = self.block_4(data_block3) + data_block3

        feature_map = self.conv_5(data_block4)

        feature_size = feature_map.shape[-1]
        
        props = proposals.view(-1, 5).cpu().numpy()

        idxs_start =  np.floor(props[:, 0] / props[:, 2] * feature_size).astype(np.int16)
        idxs_end = np.floor(props[:, 1] / props[:, 2] * feature_size).astype(np.int16) + 1
        
        # roi_pooling
        #--------------ROI Pooling in 1D-------------#

        roi_features = []
        for i in range(len(idxs_start)):
            roi_features.append(self.roi_pool(feature_map[:, :, idxs_start[i]:idxs_end[i] + 1]))
        
        roi_features = torch.cat(roi_features)


        header_data = self.header_conv3(self.header_conv2(self.header_conv1(roi_features)))

        data_fc = header_data.view(-1, 1, 5*128)

        loc_pred = self.loc_fc_3((self.loc_fc_2(self.leaky_relu(self.loc_fc_1(data_fc)))))

        cls_pred = self.cls_fc_3((self.cls_fc_2(self.leaky_relu(self.cls_fc_1(data_fc)))))

        return loc_pred, cls_pred
