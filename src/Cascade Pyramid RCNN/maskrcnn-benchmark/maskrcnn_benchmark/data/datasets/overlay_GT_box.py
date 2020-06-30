import random
import numpy as np
from maskrcnn_benchmark.engine.bBox_2D import bBox_2D
import math
from torchvision import transforms
import torch
import cv2


def overlay_GT_on_scan(img, anno, cloudgt, anngt, resolution=1000):
    #  one img and its ann (ann in pixels)
    #  GT database of cloudgt and anngt (cloud and ann in original METERs !!!)
    #  thus, box.resize is needed
    point_collide = 0
    box_collide = 0

    if len(anno) == 0:
        return img, anno

    ann_num = len(anno)
    sampling_num = int(max(0, (15 - ann_num)))  # 12 is the max box num in this dataset
    # if ann_num <= 5:
    #     sampling_num = int(random.random() * max(0, (15 - ann_num)))  # 12 is the max box num in this dataset
    # else:
    #     return img, anno
    collision_box = False  # indicator for collision test
    collision_point = False
    _pixel_enhance = np.array([-1, 0, 1])
    pixel_enhance = np.array([[x, y] for x in _pixel_enhance for y in _pixel_enhance])  # enhance pixel by extra 8

    existing_box_vertex = []
    for ann in anno:  # for every existing GT box
        label = ann["bbox"]
        orien = ann["rotation"]
        ebox = bBox_2D(label[3], label[2], label[0] + label[2] / 2, label[1] + label[3] / 2, orien)
        # bBox_2D: length, width, xc, yc,alpha       label: 'bbox': [box.xtl, box.ytl, box.width, box.length],
        # ebox.resize(1.2)
        ebox.bBoxCalcVertxex()
        existing_box_vertex.append([ebox.vertex1, ebox.vertex2, ebox.vertex3, ebox.vertex4])  # shape [N,4,2]
    existing_box_vertex = np.array(existing_box_vertex).reshape(-1, 2)

    img = transforms.ToTensor()(img)
    img = img.permute(1, 2, 0)
    for num in range(sampling_num):  # for every sampled GT box
        index = int(random.random() * len(anngt))
        ann_sampled = anngt[index]
        point_sampled = cloudgt[index]
        box = bBox_2D(ann_sampled['bbox'][3], ann_sampled['bbox'][2], ann_sampled['bbox'][0], ann_sampled['bbox'][1],
                      ann_sampled['rotation'])
        box.scale(900 / 30, 500, 100)
        # box.scale(resolution / 200, 0, 0)  # converse to img pixel values
        # box.resize(1.2)  # to ensure points inside
        box.bBoxCalcVertxex()
        box.xcyc2topleft()
        maxx = max(box.vertex1[0], box.vertex2[0], box.vertex3[0], box.vertex4[0])
        minx = min(box.vertex1[0], box.vertex2[0], box.vertex3[0], box.vertex4[0])
        maxy = max(box.vertex1[1], box.vertex2[1], box.vertex3[1], box.vertex4[1])
        miny = min(box.vertex1[1], box.vertex2[1], box.vertex3[1], box.vertex4[1])

        # collision test first
        for line in img[miny:maxy, :, :]:
            breakflag = False
            for pixel in line[minx:maxx, :]:  # !!!  y -x !!!!!!!
                if pixel[0] != 0:
                    breakflag = True
                    break
            if breakflag:
                collision_point = True
                point_collide += 1
                break

        for vertex in existing_box_vertex:
            breakflag = False
            if minx < vertex[0] < maxx and miny < vertex[1] < maxy:
                collision_box = False
                box_collide += 1
                break
            else:
                eb = existing_box_vertex.reshape(-1, 4, 2)
                for ebox in eb:
                    maxxeb = max(ebox[0][0], ebox[1][0], ebox[2][0], ebox[3][0])
                    minxeb = min(ebox[0][0], ebox[1][0], ebox[2][0], ebox[3][0])
                    maxyeb = max(ebox[0][1], ebox[1][1], ebox[2][1], ebox[3][1])
                    minyeb = min(ebox[0][1], ebox[1][1], ebox[2][1], ebox[3][1])
                    if (minxeb < box.vertex1[0] < maxxeb and minyeb < box.vertex1[1] < maxyeb) or \
                            (minxeb < box.vertex2[0] < maxxeb and minyeb < box.vertex2[1] < maxyeb) or \
                            (minxeb < box.vertex3[0] < maxxeb and minyeb < box.vertex3[1] < maxyeb) or \
                            (minxeb < box.vertex4[0] < maxxeb and minyeb < box.vertex4[1] < maxyeb):
                        breakflag = True
                        break
                if breakflag:
                    collision_box = True
                    box_collide += 1
                    break

        # if collision_box or collision_point:  # for testing
        #     for dot in point_sampled:
        #         if dot[1] < 30 and 100 / 6 > dot[0] > -100 / 6:  # in range
        #             x, y = dot[1] * 180 / 30 + 20, dot[0] * 6 + 100
        #             x = int(x * resolution / 200)
        #             y = int(y * resolution / 200)
        #             enhanced = [[x, y] + e for e in pixel_enhance]
        #             for e in enhanced:
        #                 if e[0] < resolution and 0 <= e[0] and e[1] < resolution and 0 <= e[0]:
        #                     if collision_point and not collision_box:
        #                         pp = torch.FloatTensor([1, 0, 0])
        #                     if collision_box and not collision_point:
        #                         pp = torch.FloatTensor([1, 1, 0])
        #                     if collision_point and collision_box:
        #                         pp = torch.FloatTensor([1, 1, 1])
        #                     img[e[0], e[1]] = pp
        #     anninfo = {
        #         'segmentation': [],
        #         'area': box.length * box.width,
        #         'image_id': anno[0]['image_id'],
        #         'bbox': [box.xtl, box.ytl, box.width, box.length],
        #         'rotation': box.alpha,
        #         'category_id': 1,
        #         'id': 999,
        #         'iscrowd': 0
        #     }
        #     anno.append(anninfo)
        #
        #     # box.resize(1 / 1.2)
        #     box.bBoxCalcVertxex()
        #     new_box_vertex = np.array([box.vertex1, box.vertex2, box.vertex3, box.vertex4]).reshape(-1, 2)
        #     existing_box_vertex = np.concatenate((existing_box_vertex, new_box_vertex))

        if not (collision_box or collision_point):
            for dot in point_sampled:
                if dot[1] < 30 and 100 / 6 > dot[0] > -100 / 6:  # in range
                    x, y = dot[1] * 900 / 30 + 100, dot[0] * 30 + 500
                    x = int(x)
                    y = int(y)
                    enhanced = [[x, y] + e for e in pixel_enhance]
                    for e in enhanced:
                        if e[0] < resolution and 0 <= e[0] and e[1] < resolution and 0 <= e[0]:
                            pp = torch.FloatTensor([
                                # int(255 - math.hypot(dot[1], dot[0]) * 255 / 60) / 255,
                                # int(255 - (dot[1] * 235 / 30 + 20)) / 255,
                                # int(dot[0] * 75 / 15 + 80) / 255
                                # 0.4235, 0.9294, 1.0000
                                1, 1, 1
                                # ?????????  how to calculate this RGB color encoding values ????
                            ])
                            img[e[0], e[1]] = pp

            anninfo = {
                'segmentation': [],
                'area': box.length * box.width,
                'image_id': anno[0]['image_id'],
                'bbox': [box.xtl, box.ytl, box.width, box.length],
                'rotation': box.alpha,
                'category_id': 1,
                'id': 999,
                'iscrowd': 0
            }
            anno.append(anninfo)

            # box.resize(1 / 1.2)
            box.bBoxCalcVertxex()
            new_box_vertex = np.array([box.vertex1, box.vertex2, box.vertex3, box.vertex4]).reshape(-1, 2)
            existing_box_vertex = np.concatenate((existing_box_vertex, new_box_vertex))

    img = img.permute(2, 0, 1)
    img = transforms.ToPILImage()(img)
    # print(point_collide, box_collide)
    return img, anno
