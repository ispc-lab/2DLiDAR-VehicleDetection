import os
# import pcl
import torch
import numpy as np
import cv2
import math
from bBox_2D import bBox_2D
import json
import random
import shutil

cloudata_train = np.load('./testset/cloudata_train_new.npy')
cloudata_test = np.load('./testset/cloudata_test_new.npy')
anndata_train = np.load('./testset/anndata_train_new.npy')
anndata_test = np.load('./testset/anndata_test_new.npy')
# time_stamps = joblib.load('./testset/time_stamps_tmp')

# ==============================
resolution = 1000  # res*res !!!   (224 ResNet  299 Inception  1000 Visualization ONLY)


# ==============================
# ------------>   x    annotation box clock wise
# |
# |
# |
# y
# ==============================

def data2image(cloudata, anndata, isTrain):
    img = []
    # Cloud data to images
    # _pixel_enhance = np.array([-1, 0, 1])
    # pixel_enhance = np.array([[x, y] for x in _pixel_enhance for y in _pixel_enhance])  # enhance pixel by extra 8
    _pixel_enhance = np.array([-1, 0, 1])
    pixel_enhance = np.array([[x, y] for x in _pixel_enhance for y in _pixel_enhance])  # enhance pixel by extra 8
    for i, scan in enumerate(cloudata):
        emptyImage = np.zeros([1000, 1000, 3], np.uint8)
        for dot in scan:
            if dot[0] < 30 and 100 / 6 > dot[1] > -100 / 6:  # in range
                x, y = int(dot[0] * 900 / 30 + 100), int(dot[1] * 30 + 500)
                enhanced = [[x, y] + e for e in pixel_enhance]
                for e in enhanced:
                    if e[0] < 1000 and 0 <= e[0] and e[1] < 1000 and 0 <= e[0]:
                        emptyImage[e[0], e[1]] = (
                            int(255 - math.hypot(dot[0], dot[1]) * 255 / 60),
                            int(255 - (dot[0] * 235 / 30 + 20)),
                            int(dot[1] * 75 / 15 + 80)
                            # 255, 255, 255
                        )

        outImage = cv2.resize(emptyImage, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

        for j, label in enumerate(anndata[i]):
            if label[4] == -90 or label[4] == 90:
                box = bBox_2D(label[1], label[0], label[3], label[2],
                              -label[4])  # fix annotations!!!   x-y from data is reversed
            else:
                box = bBox_2D(label[0], label[1], label[3], label[2], -label[4])  # clock wise

            # print(box.xc,box.yc)
            if box.xc == 0 and box.yc == 0 and box.length == 0 and box.width == 0:
                anndata[i][j] = [0, 0, 0, 0, 0]  # mark with 0
                continue
            # print(' xc ', box.xc, ' yc ', box.yc, ' l ', box.length, ' w ', box.width)
            box.scale(900 / 30, 500, 100)
            box.scale(resolution / 1000, 0, 0)

            anndata[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]

        #     rad = box.alpha * math.pi / 180
        #     box.bBoxCalcVertxex()
        #     cv2.line(outImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
        #     cv2.line(outImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
        #     cv2.line(outImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
        #     cv2.line(outImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)
        #     point = int(box.xc - box.length * 0.8 * np.sin(rad)), int(box.yc + box.length * 0.8 * np.cos(rad))
        #     cv2.line(outImage, (int(box.xc), int(box.yc)),
        #              point,
        #              (155, 255, 255), 1, cv2.LINE_AA)
        #     print(box.xc, box.yc,box.alpha)
        # cv2.imshow('scan', outImage)
        print(i)
        # k=cv2.waitKey()
        # if k == 27:  # Esc for exiting
        #     cv2.destroyAllWindows()
        #     os._exit(1)

        img.append(outImage)

    # Flipping
    if isTrain:
        augmentimg = []
        for i, im in enumerate(img):
            imflipped = cv2.flip(im, 1)
            augmentimg.append(imflipped)
        img = img + augmentimg
        del augmentimg

        augmentann = np.zeros(anndata.shape, dtype=np.float)
        for i, scan in enumerate(anndata):
            for j, label in enumerate(scan):
                if label[0] == 0:
                    continue
                box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
                box.flipx(axis=int(resolution / 2))
                augmentann[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]
        anndata = np.concatenate((anndata, augmentann))
        del augmentann
    #
    # Adding noise : rotate, translate(x,y), resize
    # print('Adding Noise...')
    # augmentann = np.zeros(anndata.shape, dtype=np.float)
    # for i, scan in enumerate(anndata):
    #     for j, label in enumerate(scan):
    #         if label[0]==0:
    #             continue
    #         noiseratio = ((torch.randn(2)).div_(20)).exp_()
    #         noiseoffset = (torch.randn(2))
    #         box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
    #         box.rotate(noiseratio[0])
    #         box.resize(noiseratio[1])
    #         box.translate(noiseoffset[0], noiseoffset[1])
    #         augmentann[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]
    # anndata = np.concatenate((anndata, augmentann))
    # del augmentann
    # img = img + img
    # # #

    ll = len(img)
    print(anndata.shape, '\t', ll)

    return img, anndata


# to COCO json dataset and shuffle and split
# ann_json = {}
# images = []
# annotations = []
# categories = []
# iminfo = {}
# anninfo = {}
# catinfo = {}
# trainsplit, valsplit, testsplit = int(ll * 0.70), int(ll * (0.70 + 0.15)), ll
# overfittest = 60
# print(trainsplit, valsplit - trainsplit, testsplit - valsplit)
# mwidth, mlength, mrotation, marea = 0, 0, 0, 0
#
# shutil.rmtree('./maskrcnn-benchmark/datasets/coco/val2014')
# os.mkdir('./maskrcnn-benchmark/datasets/coco/val2014')
# shutil.rmtree('./maskrcnn-benchmark/datasets/coco/train2014')
# os.mkdir('./maskrcnn-benchmark/datasets/coco/train2014')
# shutil.rmtree('./maskrcnn-benchmark/datasets/coco/test2014')
# os.mkdir('./maskrcnn-benchmark/datasets/coco/test2014')
# shutil.rmtree('./maskrcnn-benchmark/datasets/coco/overfit2014')
# os.mkdir('./maskrcnn-benchmark/datasets/coco/overfit2014')  # renew data space

def create_coco_train(img, anndata):
    images = []
    annotations = []
    categories = []
    mwidth, mlength, mrotation, marea = 0, 0, 0, 0
    shutil.rmtree('./maskrcnn-benchmark/datasets/coco/train2014')
    os.mkdir('./maskrcnn-benchmark/datasets/coco/train2014')

    pixel_mean = np.array([0., 0., 0.])
    pixel_std = np.array([0., 0., 0.])
    for i, im in enumerate(img):
        cv2.imwrite('./maskrcnn-benchmark/datasets/coco/train2014/im%d.jpg' % i, im)
        pixel_mean += np.array([np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])])
        pixel_std += np.array([np.std(im[:, :, 0]), np.std(im[:, :, 1]), np.std(im[:, :, 2])])
        iminfo = {
            "file_name": "im%d.jpg" % i,
            "height": im.shape[0],
            "width": im.shape[1],
            "id": i
        }
        images.append(iminfo)
    ll = len(img)
    print(pixel_mean / ll, '==pixel_mean==', pixel_std / ll, '==pixel_std==')

    idcount = 0
    for j, ann in enumerate(anndata):
        # np.save('./testset/dataset/ann/ann%d' % j, ann)
        for i, label in enumerate(ann):
            # remove empty
            if label[0] == 0:
                continue

            # filter bbox too small too large or too thin!! (unit in PIXELs)
            if label[0] < 12 or label[1] < 12 or label[0] > 144 or label[1] > 144 or label[0] * label[1] < 360 or label[
                0] * \
                    label[1] > 13000:
                continue

            box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
            box.xcyc2topleft()
            anninfo = {
                'segmentation': [],
                'area': box.length * box.width,
                'image_id': j,
                'bbox': [box.xtl, box.ytl, box.width, box.length],
                'rotation': box.alpha,
                'category_id': 0,
                'id': idcount,
                'iscrowd': 0
            }
            annotations.append(anninfo)
            idcount += 1
            mwidth += box.width
            mlength += box.length
            marea += box.length * box.width
            mrotation += box.alpha

    catinfo = {
        "supercategory": "none",
        "id": 0,
        "name": "car"}

    categories.append(catinfo)

    imagetrain = images[:]
    imids = set(im['id'] for im in imagetrain)
    annids = set(ann['id'] if ann['image_id'] in imids else None for ann in
                 annotations)  # get binary inds and ids of ann according to im
    # annids.remove(None)
    anntrain = []
    for ann in annotations:
        if ann['image_id'] in imids:  # two different ids !!!!!!!
            anntrain.append(ann)
    trainann_json = {'info': {}, 'images': imagetrain, 'annotations': anntrain, 'categories': categories}
    with open("./maskrcnn-benchmark/datasets/coco/annotations/trainann.json", 'w', encoding='utf-8') as json_file:
        json.dump(trainann_json, json_file, ensure_ascii=False)

    print('train summary ', mwidth / idcount, mlength / idcount, marea / idcount, mrotation / idcount)


def create_coco_test(img, anndata):
    images = []
    annotations = []
    categories = []
    mwidth, mlength, mrotation, marea = 0, 0, 0, 0
    shutil.rmtree('./maskrcnn-benchmark/datasets/coco/test2014')
    os.mkdir('./maskrcnn-benchmark/datasets/coco/test2014')

    pixel_mean = np.array([0., 0., 0.])
    pixel_std = np.array([0., 0., 0.])
    for i, im in enumerate(img):
        cv2.imwrite('./maskrcnn-benchmark/datasets/coco/test2014/im%d.jpg' % i, im)
        pixel_mean += np.array([np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])])
        pixel_std += np.array([np.std(im[:, :, 0]), np.std(im[:, :, 1]), np.std(im[:, :, 2])])
        iminfo = {
            "file_name": "im%d.jpg" % i,
            "height": im.shape[0],
            "width": im.shape[1],
            "id": i
        }
        images.append(iminfo)
    ll = len(img)
    print(pixel_mean / ll, '==pixel_mean==', pixel_std / ll, '==pixel_std==')

    idcount = 0
    for j, ann in enumerate(anndata):
        # np.save('./testset/dataset/ann/ann%d' % j, ann)
        for i, label in enumerate(ann):
            # remove empty
            if label[0] == 0:
                continue

            # filter bbox too small too large or too thin!! (unit in PIXELs)
            if label[0] < 12 or label[1] < 12 or label[0] > 144 or label[1] > 144 or label[0] * label[1] < 360 or label[
                0] * \
                    label[1] > 13000:
                continue

            box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
            box.xcyc2topleft()
            anninfo = {
                'segmentation': [],
                'area': box.length * box.width,
                'image_id': j,
                'bbox': [box.xtl, box.ytl, box.width, box.length],
                'rotation': box.alpha,
                'category_id': 0,
                'id': idcount,
                'iscrowd': 0
            }
            annotations.append(anninfo)
            idcount += 1
            mwidth += box.width
            mlength += box.length
            marea += box.length * box.width
            mrotation += box.alpha

    catinfo = {
        "supercategory": "none",
        "id": 0,
        "name": "car"}

    categories.append(catinfo)

    imagetest = images[:]
    imids = set(im['id'] for im in imagetest)
    annids = set(ann['id'] if ann['image_id'] in imids else None for ann in
                 annotations)  # get binary inds and ids of ann according to im
    # annids.remove(None)
    anntest = []
    for ann in annotations:
        if ann['image_id'] in imids:  # two different ids !!!!!!!
            anntest.append(ann)
    testann_json = {'info': {}, 'images': imagetest, 'annotations': anntest, 'categories': categories}
    with open("./maskrcnn-benchmark/datasets/coco/annotations/testann.json", 'w', encoding='utf-8') as json_file:
        json.dump(testann_json, json_file, ensure_ascii=False)

    print('test summary ', mwidth / idcount, mlength / idcount, marea / idcount, mrotation / idcount)


# random.shuffle(images)

# trainset
# imagetrain = images[:trainsplit]
# imids = set(im['id'] for im in imagetrain)
# annids = set(ann['id'] if ann['image_id'] in imids else None for ann in
#              annotations)  # get binary inds and ids of ann according to im
# annids.remove(None)
# anntrain = []
# for ann in annotations:
#     if ann['image_id'] in imids:  # two different ids !!!!!!!
#         anntrain.append(ann)
# trainann_json = {'info': {}, 'images': imagetrain, 'annotations': anntrain, 'categories': categories}
# with open("./maskrcnn-benchmark/datasets/coco/annotations/trainann.json", 'w', encoding='utf-8') as json_file:
#     json.dump(trainann_json, json_file, ensure_ascii=False)

# valset
# imageval = images[trainsplit:valsplit]
# imids = set(im['id'] for im in imageval)
# annids = set(ann['id'] if ann['image_id'] in imids else None for ann in
#              annotations)  # get binary inds and ids of ann according to im
# annids.remove(None)
# annval = []
# for ann in annotations:
#     if ann['image_id'] in imids:  # two different ids !!!!!!!
#         annval.append(ann)
# valann_json = {'info': {}, 'images': imageval, 'annotations': annval, 'categories': categories}
# with open("./maskrcnn-benchmark/datasets/coco/annotations/valann.json", 'w', encoding='utf-8') as json_file:
#     json.dump(valann_json, json_file, ensure_ascii=False)
#
# # testset
# imagetest = images[valsplit:]
# imids = set(im['id'] for im in imagetest)
# annids = set(ann['id'] if ann['image_id'] in imids else None for ann in
#              annotations)  # get binary inds and ids of ann according to im
# annids.remove(None)
# anntest = []
# for ann in annotations:
#     if ann['image_id'] in imids:  # two different ids !!!!!!!
#         anntest.append(ann)
# testann_json = {'info': {}, 'images': imagetest, 'annotations': anntest, 'categories': categories}
# with open("./maskrcnn-benchmark/datasets/coco/annotations/testann.json", 'w', encoding='utf-8') as json_file:
#     json.dump(testann_json, json_file, ensure_ascii=False)
#
# # overfitset
# imageoverfit = images[:overfittest]
# imids = set(im['id'] for im in imageoverfit)
# annids = set(ann['id'] if ann['image_id'] in imids else None for ann in
#              annotations)  # get binary inds and ids of ann according to im
# annids.remove(None)
# annoverfit = []
# for ann in annotations:
#     if ann['image_id'] in imids:  # two different ids !!!!!!!
#         annoverfit.append(ann)
# overfitann_json = {'info': {}, 'images': imageoverfit, 'annotations': annoverfit, 'categories': categories}
# with open("./maskrcnn-benchmark/datasets/coco/annotations/overfit.json", 'w', encoding='utf-8') as json_file:
#     json.dump(overfitann_json, json_file, ensure_ascii=False)
#
# # summary
# print(mwidth / idcount, mlength / idcount, marea / idcount, mrotation / idcount)
# #  12.588   5.719   131.970   0.0
#
#
# for im in overfitann_json['images']:
#     shutil.copyfile('./maskrcnn-benchmark/datasets/coco/train2014/' + im["file_name"],
#                     './maskrcnn-benchmark/datasets/coco/overfit2014/' + im["file_name"])
#
# for im in valann_json['images']:
#     shutil.move('./maskrcnn-benchmark/datasets/coco/train2014/' + im["file_name"],
#                 './maskrcnn-benchmark/datasets/coco/val2014/' + im["file_name"])
#
# for im in testann_json['images']:
#     shutil.move('./maskrcnn-benchmark/datasets/coco/train2014/' + im["file_name"],
#                 './maskrcnn-benchmark/datasets/coco/test2014/' + im["file_name"])

def preProcessing():
    img, ann = data2image(cloudata_train, anndata_train, isTrain=True)
    create_coco_train(img, ann)
    img, ann = data2image(cloudata_test, anndata_test, isTrain=False)
    create_coco_test(img, ann)


if __name__ == "__main__":
    preProcessing()
