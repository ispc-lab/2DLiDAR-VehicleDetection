# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from maskrcnn_benchmark.engine.bBox_2D import bBox_2D
import math
import numpy as np
from maskrcnn_benchmark.data.datasets.overlay_GT_box import overlay_GT_on_scan
import cv2
from PIL import Image


# ==============================
# ------------>   x    annotation box clock wise
# |
# |
# |
# y
# ==============================

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        # print(self.ids.__len__(),'====================================',ann_file)
        # self.gtcloud = np.load('../../testset/GTpoints.npy')
        # self.gtann = np.load('../../testset/GTanns.npy')

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        img_original = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img_original = img_original + 127.5
        trans1 = torchvision.transforms.ToTensor()
        img_original = trans1(img_original)
        # cv2.imwrite('d.jpg', img_original)
        # print('============')
        # pass

        # img, anno = overlay_GT_on_scan(img, anno, self.gtcloud, self.gtann, resolution=1000)
        #
        # # noiseoffset = (torch.randn(2))  # minimal bbox noise is better?
        # for ann in anno:
        #     noiseratio = ((torch.randn(1)).div_(20)).exp_().clamp(0.9, 1.1)
        #     noiserotate = torch.randn(1).clamp(-3, 3)
        #     label = ann["bbox"]
        #     orien = ann["rotation"]
        #     box = bBox_2D(label[3], label[2], label[0] + label[2] / 2, label[1] + label[3] / 2,
        #                   orien)  # bBox_2D: length, width, xc, yc,alpha       label: 'bbox': [box.xtl, box.ytl, box.width, box.length],
        #     box.rotate(noiserotate)
        #     box.resize(noiseratio)
        #     # box.translate(noiseoffset[0], noiseoffset[1])
        #     box.xcyc2topleft()
        #     ann["bbox"] = [box.xtl, box.ytl, box.width, box.length]
        #     # slightly stretch the box may be better viewed ?
        #     ann["rotation"] = box.alpha

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno]  # if obj["iscrowd"] == 0] ===============================================

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        # print(boxes)
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")  # =====================================

        # print(target.bbox,'============================')

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        # ====================================

        rotations = [obj["rotation"] * math.pi / 180 for obj in anno]
        # print(rotations,'====')
        rotations = torch.tensor(rotations)
        rotations = torch.stack((5 * torch.sin(rotations), 5 * torch.cos(rotations)))
        # rotations = torch.stack((rotations, rotations))  # for testing
        # COMPLEX space   *5 is radius of unit circle or weight
        rotations = torch.transpose(rotations, dim0=0, dim1=-1)  # N*2 shape
        # print(rotations)
        target.add_field("rotations", rotations)

        # print(target.get_field('rotations'), '============ooo================')

        # print(target,'============================================')
        target = target.clip_to_image(remove_empty=False)
        # print(len(target), '==================targetanno=================')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # print(img.size(),'=================%d=================='%idx)
        # print(target.get_field('rotations'), '============================')
        return img, target, idx, img_original

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
