import pcl
import os
import copy
import torch
import numpy as np

from sklearn.cluster import DBSCAN

def pointcloud_crop(pcd_inputs, anns, up_bound=None, down_bound=0, lateral_bound=None):
    """
    Crop out sepecific aera points clounds
    down_bound < x < up_bound, abs(y) < lateral_bound 
    """
    croped_points = copy.deepcopy(pcd_inputs)
    croped_anns = copy.deepcopy(anns)

    if (up_bound and lateral_bound) is not None:

        flag1 = np.logical_and(down_bound < croped_points[:, 0], croped_points[:, 0] < up_bound)                           
        flag1 = np.logical_and(flag1, croped_points[:, 1] < lateral_bound)
        flag1 = np.logical_and(flag1, croped_points[:, 1] > (-1 * lateral_bound))
        croped_points = croped_points[flag1]

        if anns is not None and len(anns) > 0:
            flag2 = np.logical_and(down_bound < croped_anns[:, 2], croped_anns[:, 2] < up_bound)
            flag2 = np.logical_and(flag2, croped_anns[:, 3] < lateral_bound)
            flag2 = np.logical_and(flag2, croped_anns[:, 3] > (-1 * lateral_bound))
            croped_anns = croped_anns[flag2]
   
    return croped_points, croped_anns

def pointcloud_augmentation_all(pcd_inputs, anns, flip_flag=1):
    """
    To augment the input points clouds and the annotations.
    The input clouds and anns are np.array types
    """

    pcd_points = copy.deepcopy(pcd_inputs)
    annotations = copy.deepcopy(anns)

    if np.random.random() > 0.5:

        ones = np.ones((len(pcd_points), 1))
        pcd_points = np.c_[pcd_points, ones]
        
        if flip_flag:
            # augment the points clouds and annotations with random rotation and random translation
            theta = (2 * np.random.random() - 1) * 10 / 180 * np.pi # control the ratotaion margin
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            tx = np.random.random()
            ty = np.random.random()
            transform_matrix = np.array([[cos_theta,  sin_theta],
                                [-sin_theta, cos_theta],
                                [tx,             ty]])
            pcd_points = np.dot(pcd_points, transform_matrix)
            if annotations is not None:
                for ann in annotations:
                    ann[2] = ann[2]*cos_theta - ann[3]*sin_theta + tx
                    ann[3] = ann[2]*sin_theta + ann[3]*cos_theta + ty
                    ann[4] += theta/3.14159*180
        else:
            # augment the points clouds and annotations with flip and random translation
            tx = np.random.random()
            transform_matrix = np.array([[1,  0],
                                        [0, -1],
                                        [tx,  0]])
            pcd_points = np.dot(pcd_points, transform_matrix)
            if annotations is not None:
                for ann in annotations:
                    ann[2] += tx
                    ann[3] *= -1
                    ann[4] *= -1
    
        return pcd_points, annotations

    else:
        # augment the points clouds with random noise
        random_noise = (np.random.random((pcd_inputs.shape)) - 0.5) * 0.05
        pcd_points += random_noise
        
        return pcd_points, annotations

def pointcloud_augmentation_individual_1(pcd_inputs, pcd_props):
    """
    augment the each pcd_props points cloud with random add or delete some points.
    """
    points_new = []
    props_new = copy.deepcopy(pcd_props)
    last_start, last_end = 0, 0
    
    points_num = len(pcd_inputs)

    if np.random.rand() > 0.5:
        for prop_idx, prop in enumerate(pcd_props):
            prop_start, prop_end = int(prop[0]), int(prop[1])
            prop_points = pcd_inputs[prop_start:(prop_end + 1), :]
            #  Add outliers
            prop_outliers = pcd_inputs[(last_end + 1):prop_start, :]
            points_new.append(prop_outliers)
            last_end = prop_end

            if np.random.rand() > 0.5 and len(prop_points) > 15:
                # random delete some points
                aug_num = 4 + int(len(prop_points)/10)
                num_rest = len(prop_points) - aug_num
                aug_prop_idx = np.random.choice(np.arange(len(prop_points)),size=num_rest, replace=False)
                aug_prop_points = prop_points[aug_prop_idx, :]
                points_new.append(aug_prop_points)
            else:
                # random add some points
                augnum = np.random.randint(3, 5)
                prop_rho = np.sqrt(prop_points[:, 0]**2 + prop_points[:, 1]**2)
                prop_theta = np.arctan2(prop_points[:, 1], prop_points[:, 0])
                # head 
                rho_0 = np.mean(prop_rho[0:3])
                aug_theta_0 = prop_theta[0] + np.arange(1, 4)*np.pi/180/2
                aug_points_0 = np.vstack((rho_0*np.sin(aug_theta_0), rho_0*np.cos(aug_theta_0))).T
                # tail
                rho_1 = np.mean(prop_rho[-3:-1])
                aug_theta_1 = prop_theta[1] - np.arange(1, 4)*np.pi/180/2
                aug_points_1 = np.vstack((rho_1*np.sin(aug_theta_1), rho_1*np.cos(aug_theta_1))).T
                aug_prop_points = np.vstack((aug_points_0, prop_points, aug_points_1))
                points_new.append(aug_prop_points)
            
        prop_outliers = pcd_inputs[(last_end + 1):prop_start, :]
        points_new.append(prop_outliers)
        # update props
        points_cur_num = 0
        for i in range(len(pcd_props)):
            points_cur_num += len(points_new[i*2])
            props_new[i][0] = points_cur_num
            points_cur_num += len(points_new[i*2 + 1])
            props_new[i][1] = points_cur_num
        points_cur_num += len(points_new[-1])
        props_new[:, -1] = points_cur_num

        pcd_points = np.vstack(points_new)
        pcd_props = props_new

        pcd_points += (np.random.random(pcd_points.shape) - 0.5) * 0.05 
    else:
        pcd_points = pcd_inputs + (np.random.random(pcd_inputs.shape) - 0.5) * 0.05 

    return pcd_points, pcd_props

def pointcloud_augmentation_individual_2(pcd_points, pcd_props, pcd_gts):
    """
    augment the each pcd_props points cloud with random rotation and translation.
    """
    for prop_idx, prop in enumerate(pcd_props):
        if np.random.rand() > 0.5:
            continue
        # update prop_points
        prop_start, prop_end = int(prop[0]), int(prop[1])+1
        prop_points = pcd_points[prop_start:prop_end, :]
        ones = np.ones((len(prop_points), 1))
        
        prop_points = np.c_[prop_points, ones]

        theta = (2*np.random.random()-1)*10/180*np.pi 
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        tx = np.random.random()
        ty = np.random.random()
        rotation_matrix = np.array([[cos_theta,  -sin_theta],
                            [sin_theta, cos_theta]])
        transform_matrix = np.array([[cos_theta,  -sin_theta],
                            [sin_theta, cos_theta],
                            [tx,             ty]])
        prop_points = np.dot(prop_points, transform_matrix)
        pcd_points[prop_start:prop_end, :] = prop_points
       

        # update prop_gt
        pcd_gts[prop_idx, 1:3] = np.dot(rotation_matrix, pcd_gts[prop_idx, 1:3])
        pcd_gts[prop_idx, -4] += theta
        pcd_gts[prop_idx, 3] = np.cos(pcd_gts[prop_idx, -4])
        pcd_gts[prop_idx, 4] = np.sin(pcd_gts[prop_idx, -4])
    
    return pcd_points, pcd_gts


def pointcloud_get_gts(points, props, anns, dis_thresh=2.25):
    
    prop_num = len(props)

    gts = np.zeros((prop_num, 13))
    gts[:, -1] = -1
    
    if anns is None or len(anns) == 0:
        return gts
    else:
        ann_num = len(anns)
    
    ann_flag = np.zeros(ann_num)
    ann_idx = np.arange(ann_num)
    dis_log = np.zeros(prop_num)
    dis_idx = np.arange(prop_num)
    prop_tag = np.zeros(prop_num)

    for idx, prop in enumerate(props):

        prop_start, prop_end = int(prop[0]), int(prop[1])
        
        prop_points = points[prop_start:prop_end, :]
        prop_points = np.reshape(prop_points, (-1, 2))

        x_c, y_c = np.mean(prop_points, axis=0)

        ann_xy = anns[:, 2:4] - np.array([x_c, y_c])
        dis = np.linalg.norm(ann_xy, axis=1)
        dis = dis.reshape(ann_num)
        dis_log[idx] = min(dis)
        gt_idx = np.argmin(dis) if (dis_log[idx] < dis_thresh) else -1
        
        if gt_idx >= 0:

            gt_length_ratio = 1
            gt_width_ratio = 1
           
            # get relative translation
            gt_dx = ann_xy[gt_idx, 0]
            gt_dy = ann_xy[gt_idx, 1]
        
            # get theta 
            gt_theta = anns[gt_idx, 4]/180.0*np.pi # theta [-pi, pi]
            gt_theta_cos = np.cos(gt_theta)
            gt_theta_sin = np.sin(gt_theta)
            
            gt_abs_theta = np.arctan2(anns[gt_idx, 3], anns[gt_idx, 2])

            gt_rho = np.sqrt(anns[gt_idx, 2]**2 + anns[gt_idx, 3]**2)

            gt_0 = [1, 
                    gt_dx, gt_dy,
                    gt_theta_cos, gt_theta_sin,
                    gt_length_ratio, gt_width_ratio,
                    ann_xy[gt_idx, 0], ann_xy[gt_idx, 1],
                    gt_theta,
                    gt_abs_theta,
                    gt_rho,
                    gt_idx
                    ]

            gts[idx,:] = gt_0

            ann_flag[gt_idx] += 1
            prop_tag[idx] = gt_idx

    return gts

def pointcloud_get_props(pcd_points, eps=0.8, min_samp=5):
    if len(pcd_points) < min_samp:
        min_samp = len(pcd_points)
    if len(pcd_points) < 10:
        return None
    dbscan = DBSCAN(eps=eps, min_samples=min_samp, algorithm='ball_tree').fit(pcd_points)
    labels = dbscan.labels_
    cluster_nums = len(set(labels)) - (1 if -1 in labels else 0)
    
    point_num  = len(pcd_points)
    point_idxs = np.arange(point_num)
    pcd_props = []
    for cluster_i in range(cluster_nums):
        cluster_indices = (np.full_like(point_idxs, cluster_i) == labels)
        if point_idxs[cluster_indices][-1] -  point_idxs[cluster_indices][0] < min_samp:
            continue
        x_c, y_c = np.mean(pcd_points[cluster_indices,:], axis=0)
        prop = [point_idxs[cluster_indices][0], point_idxs[cluster_indices][-1], point_num, x_c, y_c]
        # print(prop)
        pcd_props.append(prop)
    
    pcd_props = np.array(pcd_props)
    arg_idx = np.argsort(pcd_props[:, 0], axis=0)

    pcd_props = pcd_props[arg_idx, :]

    return pcd_props

def pointcloud_get_input_data(pcd_points, pcd_props):

    input_data = copy.deepcopy(pcd_points)
    d_x = copy.deepcopy(input_data[:, 0]).reshape(-1, 1)
    d_y = copy.deepcopy(input_data[:, 1]).reshape(-1, 1)
    x = copy.deepcopy(input_data[:, 0]).reshape(-1, 1)
    y = copy.deepcopy(input_data[:, 1]).reshape(-1, 1)
    prop_num = len(pcd_props)
  
    for prop_idx, prop in enumerate(pcd_props):
        prop_start, prop_end = int(prop[0]), int(prop[1])
        prop_points = pcd_points[prop_start:(prop_end + 1), :]
        x_c, y_c = np.mean(prop_points, axis=0)
    
        d_x[prop_start:(prop_end + 1)] -= x_c
        d_y[prop_start:(prop_end + 1)] -= y_c

        prop_length = np.max(d_x[prop_start:(prop_end + 1)]) - np.min(d_x[prop_start:(prop_end + 1)])
        prop_width  = np.max(d_y[prop_start:(prop_end + 1)]) - np.min(d_y[prop_start:(prop_end + 1)])


    d_rho = np.sqrt(d_x**2 + d_y**2)

    d_the = np.arctan2(d_y, d_x)

    d_cos = np.cos(d_the)
    
    d_sin = np.sin(d_the)
    
    rho = np.sqrt(x**2 + y**2)


    input_data = np.hstack((x, y, rho, d_x, d_y, d_rho))

    for prop in pcd_props:
        prop_start, prop_end = int(prop[0]), int(prop[1])
        input_data[prop_start:(prop_end + 1), 3:8] = (input_data[prop_start:(prop_end + 1), 3:8] - np.mean(input_data[prop_start:(prop_end + 1), 3:8]))\
                                                     / np.std(input_data[prop_start:(prop_end + 1), 3:8])
    input_data[:, 0:3] = (input_data[:, 0:3] - np.mean(input_data[:, 0:3])) / np.std(input_data[:, 0:3])

    return input_data