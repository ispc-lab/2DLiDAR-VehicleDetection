# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import shutil
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize

import cv2
import math
from maskrcnn_benchmark.engine.bBox_2D import bBox_2D
import numpy as np
import numpy.linalg as la
import pandas as pd
import time

# from flashtorch.utils import apply_transforms, load_image
# from flashtorch.saliency import Backprop
# from flashtorch.activmax import GradientAscent

import matplotlib.pyplot as plt
import joblib


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    eval_distance = []
    eval_angle = []
    outcenterlist = 0
    tarcenterlist = 0
    infertimelist = []
    detections_total = {}

    # model.to(cpu_device)
    # backprop = Backprop(model)
    # image = load_image('/home/wangfa/Workspace/jupiter/maskrcnn-benchmark/datasets/coco/train2014/im14.jpg')
    # input_ = apply_transforms(image)
    #
    # target_class = 24
    # backprop.visualize(input_, target_class, guided=True)
    # plt.show()

    time_stamps = joblib.load('../../testset/time_stamps_tmp')
    time_stamps_id = 0

    for i, batch in enumerate(tqdm(data_loader)):
        time_start = time.time()
        images, targets, image_ids, images_original = batch
        # print(targets,'============================')
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            time_end = time.time()
            # print(output.size(),'=========oo')
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )

        x = images_original.tensors.permute(0, 2, 3, 1)
        images_original = x.cpu().detach().numpy().copy()
        del x

        rangecoor = [540, 360, 252, 108]  # in res 600 output 608  15m 10m 7m 3m
        r = rangecoor[2]
        # ymin = int(600 - r)
        ymin = 60
        # ymax = 60 + r
        ymax = 600
        xmin = int(300 - 0.5 * r)
        xmax = int(300 + 0.5 * r)
        xylimits = [xmin, xmax, ymin, ymax]

        for j, (im, tar, out) in enumerate(zip(images_original, targets, output)):

            timestamp = time_stamps[time_stamps_id]

            outcenter, outalpha, detections_per_frame = overlay_boxes(im, out, 'output', xylimits, 1000)
            tarcenter, taralpha, _ = overlay_boxes(im, tar, 'targets', xylimits, 1000)

            detections_total[timestamp] = detections_per_frame
            # np.savetxt('./detections/' + timestamp + '.txt', detections_per_frame,
            #            fmt='%.4f', delimiter='\t', newline='\n')
            # cv2.imwrite('./result/%d' % i + '_' + timestamp + '.jpg', im)
            time_stamps_id += 1

            m, n = outcenter.shape
            o, p = tarcenter.shape
            outcenterlist += n
            tarcenterlist += p

            # cv2.imshow('scan', im)
            # print('imwritten%d'%i)
            # k=cv2.waitKey()
            # if k == 27:  # Esc for exiting
            #     cv2.destroyAllWindows()
            #     os._exit(1)

            if p == 0:
                continue
            if n == 0:
                continue
            D = np.zeros([n, p])
            A = np.zeros([n, p])
            for q in range(n):
                for j in range(p):
                    D[q, j] = la.norm(outcenter[:, q] - tarcenter[:, j])  # distance matrix
                    # A[q, j] = outalpha[q] - taralpha[j]
                    if outalpha[q] > 0 and taralpha[j] > 0 or outalpha[q] < 0 and taralpha[j] < 0:
                        A[q, j] = abs(outalpha[q] - taralpha[j])
                    else:
                        A[q, j] = abs(outalpha[q] + taralpha[j])
            for ii in range(p):
                eval_distance.append(D[np.argmin(D, axis=0)[ii]][ii])
                eval_angle.append(A[np.argmin(D, axis=0)[ii]][ii])
            # print(out.shape)

    tarnumfiltered = eval_distance.__len__()
    dislist = [0.15, 0.3, 0.45]
    anglist = [5, 15, 25, 360]
    prerec = []
    #
    for dis in dislist:
        for ang in anglist:
            predinrange = sum(
                (np.array(eval_distance) < dis * 30) & (np.array(eval_angle) < ang))  # calc matched predictions
            prednum = outcenterlist
            tarnumraw = tarcenterlist
            print(predinrange, prednum, tarnumraw, '++++', sum(np.array(eval_distance) < dis * 30),
                  sum(np.array(eval_angle) < ang))
            pre = predinrange / prednum if prednum != 0 else 1
            rec = predinrange / tarnumraw if tarnumraw != 0 else 1
            print(' precision: %.6f' % pre, ' racall: %.6f' % rec, ' with dis', dis, 'ang', ang)
            prerec.append([pre, rec, dis, ang])
    # df = pd.DataFrame(prerec)
    # df.to_csv('./prerec.csv', index=0, index_label=0)
    # df = pd.DataFrame(infertimelist)
    # df.to_csv('./infertime.csv', index=0, index_label=0)

    joblib.dump(detections_total, './detections_total')

    return results_dict, np.array(prerec)


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions, prerec = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        # torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
        pass

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args), prerec


def overlay_boxes(image, predictions, anntype, xylimits, res):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    # labels = predictions.get_field("labels")
    imgsrc = image.copy()
    oriens = predictions.get_field("rotations")
    if anntype == 'output':
        scores = predictions.get_field('scores')
    else:
        scores = oriens
    boxes = predictions.bbox
    xclist = []
    yclist = []
    alphalist = []
    detections_per_frame = []

    # print('\noriens:',oriens.size(),'boxes:',boxes.size(),'==========\n')

    for box, orien, score in zip(boxes, oriens, scores):
        color = {'targets': (155, 255, 255), 'output': (155, 255, 55)}
        offset = {'targets': 2, 'output': 0}

        box = box.squeeze_().detach().cpu().numpy()
        box_ori = box.copy()
        alpha = torch.atan2(orien[:][0], orien[:][1]) * 180 / math.pi
        # alpha = (orien[:][0] + orien[:][1]) * 0.5 * 180 / math.pi  # for testing!
        alpha = alpha.squeeze_().detach().cpu().numpy()
        # print(alpha,anntype,'====')
        if anntype == 'output':
            score.squeeze_().detach().cpu().numpy()
        # top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        top_left, bottom_right = box[:2], box[2:]
        l = bottom_right[1] - top_left[1]
        w = bottom_right[0] - top_left[0]
        xc = (top_left[0] + bottom_right[0]) / 2
        yc = (top_left[1] + bottom_right[1]) / 2

        # if not (xylimits[0] < xc < xylimits[1] and xylimits[2] < yc < xylimits[3]):
        #     # continue
        #     pass
        # cv2.line(image, (xylimits[0], xylimits[3]), (xylimits[1], xylimits[3]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[0], xylimits[2]), (xylimits[1], xylimits[2]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[0], xylimits[3]), (xylimits[0], xylimits[2]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[1], xylimits[3]), (xylimits[1], xylimits[2]), color[anntype], 1, cv2.LINE_AA)

        xclist.append(xc)
        yclist.append(yc)
        alphalist.append(alpha)

        # if l * w <= 1:
        #     continue

        box = bBox_2D(l, w, xc + offset[anntype], yc + offset[anntype], alpha)
        box.scale(res / 600, 0, 0)
        # box.resize(1.2)
        box.bBoxCalcVertxex()

        rad = box.alpha * math.pi / 180
        cv2.line(image, box.vertex1, box.vertex2, color[anntype], 2, cv2.LINE_AA)
        cv2.line(image, box.vertex2, box.vertex4, color[anntype], 2, cv2.LINE_AA)
        cv2.line(image, box.vertex3, box.vertex1, color[anntype], 2, cv2.LINE_AA)
        cv2.line(image, box.vertex4, box.vertex3, color[anntype], 2, cv2.LINE_AA)

        if anntype == 'output':
            print('+++++')
            print(box.vertex4, box.vertex3, box.vertex2, box.vertex1, '====', l * w, '\t', l, '\t', w, '\t angle',
                  box.alpha, ' score ', score)
            detections_per_frame.append([score, (box.yc - 100) / 30.0, (box.xc - 500) / 30.0, rad])
        else:
            # print(box.vertex4, box.vertex3, box.vertex2, box.vertex1, '====', l * w, '\t', l, '\t', w, '\t angle',
            # box.alpha)
            pass

        point = int(box.xc - box.length * 0.8 * np.sin(rad)), int(box.yc + box.length * 0.8 * np.cos(rad))
        cv2.line(image, (int(box.xc), int(box.yc)),
                 point,
                 color[anntype], 2, cv2.LINE_AA)
        if anntype == 'output':
            cv2.putText(image, str(score.numpy()), point, fontFace=1, fontScale=1.5, color=(255, 0, 255))

    image = cv2.addWeighted(imgsrc, 0.4, image, 0.6, 0)
    if anntype == 'output':
        detections_per_frame = np.array(detections_per_frame)
    else:
        detections_per_frame = []
    return np.array([xclist, yclist], dtype=float), np.array(alphalist, dtype=float), detections_per_frame
