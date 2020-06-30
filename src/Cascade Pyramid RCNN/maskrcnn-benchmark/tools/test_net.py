# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

import numpy as np
import os

import matplotlib.pyplot as plt
import random
from time import time
import math
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(thresh):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="file:///z/home/mbanani/nonexistant"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    # ===================================
    cfg.MODEL.ROI_HEADS.defrost()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH = thresh
    prerec = []
    # ====================================

    model = build_detection_model(cfg)

    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        a, prerec = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()

    del model
    return prerec


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def plot_pr_array(pr_array, dislist, radlist):
    pr_dict = {}

    for prerec in pr_array:
        for dis in dislist:
            for ang in radlist:
                if str(dis) + '_' + str(ang) not in pr_dict:
                    pr_dict[str(dis) + '_' + str(ang)] = []
                p_r = prerec[np.where(np.logical_and(prerec[:, 2] == dis, prerec[:, 3] == ang) == True)[0], :]
                pr_dict[str(dis) + '_' + str(ang)].append(p_r)

    for dis in dislist:
        for ang in radlist:
            curve = np.array(pr_dict[str(dis) + '_' + str(ang)])
            curve = curve.clip(max=1)
            ap = calcAP(curve) * 100
            plt.plot(curve[:, :, 1], curve[:, :, 0], color=randomcolor(),
                     label=str(dis) + '_' + str('%.2f' % ang) + '_%.1f' % ap)
            plt.xlim((0, 1))
            plt.ylim((0, 1))
    plt.legend()
    plt.xticks([x / 10.0 for x in range(11)])
    plt.yticks([x / 10.0 for x in range(11)])
    plt.grid()
    plt.savefig('./pr_curve/pr_curve' + str(time()) + '.png')


def calcAP(prcurve):
    recall = prcurve[:, :, 1]
    precision = prcurve[:, :, 0]
    precision = precision[::-1]
    recall = recall[::-1]

    acum_area = 0
    prevpr, prevre = 1, 0
    for pr, re in zip(precision, recall):
        if re[0] > prevre and (not math.isnan(pr[0])) and (not math.isnan(re[0])):
            acum_area += 0.5 * (pr[0] + prevpr) * (re[0] - prevre)
            prevpr = pr[0]
            prevre = re[0]

    return min(acum_area, 1.0)


if __name__ == "__main__":
    thresh_list = [0, 0.005, 0.1, 0.2, 0.3, 0.4, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 1]
    # thresh_list = [0.9,]

    # prerec structure per N thresh (N*12*4):
    #
    #   precision   recall  dis orien
    #                       15  5
    #                       15  15
    #                       15  25
    #                       15  360
    #                       30  5
    #                       30  15
    #                       30  25
    #                       30  360
    #                       45  5
    #                       45  15
    #                       45  25
    #                       45  360
    #

    pr_array = []
    dislist = [0.15, 0.3, 0.45]
    anglist = [5, 15, 25, 360]

    for thresh in tqdm(thresh_list):
        prerec = main(thresh)
        pr_array.append(prerec)
        print(thresh, '================================================')
    np.save('prerec', pr_array)

    # plotting pr_array
    plot_pr_array(pr_array, dislist, anglist)
