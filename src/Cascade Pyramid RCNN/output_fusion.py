import numpy as np

import joblib
import os
import math
import numpy.linalg as la
# from tools.test_net import plot_pr_array
from tqdm import tqdm
import warnings
import random
import matplotlib.pyplot as plt
from time import time

warnings.filterwarnings("ignore")

pathbox = './testset_new/test_new/bbox/'
# pathbox = './new_data/test_1/bbox/'

inputpath_1 = './maskrcnn-benchmark/tools/detections_ver2/'
inputpath_2 = './maskrcnn-benchmark/tools/detections_2nd/'

dislist = [0.15, 0.3, 0.45]
anglist = [5, 15, 25]
radlist = [ang * math.pi / 180.0 for ang in anglist]


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


def detections2prerec(thresh):
    eval_distance = []
    eval_angle = []
    prednum = 0
    tarnumraw = 0

    for fpathe, dirs, fs in os.walk(pathbox):
        for f in fs:
            # time_stamp = f.split('.')[0] + '.' + f.split('.')[1]

            # adding up to 2 dimensions for unified numpy slicing
            target = np.loadtxt(pathbox + f, usecols=[1, 2, 3, 5, 6, 9])  # coef, len, width, xc, yc, rad
            # target = np.loadtxt(pathbox + f, usecols=[x + 1 for x in range(9)])
            if len(target) > 0:
                if len(target.shape) == 1:
                    target = target[np.newaxis, :]
            # fixing annotation mistakes
            if len(target) > 0:
                target[:, 5] = -target[:, 5]
            #
            if len(target) > 0:
                target = target[np.logical_or(target[:, 3] > 0.06, target[:, 3] < -0.06)]
                target = target[np.logical_and(target[:, 3] > 0, target[:, 3] < 16.66)]
                target = target[np.logical_or(target[:, 4] < -0.06, target[:, 4] > 0.06)]
                target = target[np.logical_and(target[:, 4] < 16.66, target[:, 4] > -16.66)]
                target = target[np.logical_and(target[:, 2] > 12 / 30.0, target[:, 2] < 144 / 30.0)]
                target = target[np.logical_and(target[:, 1] > 12 / 30.0, target[:, 1] < 144 / 30.0)]
                target = target[np.logical_and(target[:, 2] * target[:, 1] > 360 / 900.0,
                                               target[:, 2] * target[:, 1] < 13000 / 900.0)]
            #
            max_target_num = 5  # 12 equals to no restriction, difference is minor
            if len(target) > 0:
                area_column = np.zeros((len(target), 1), dtype=np.float)
                data = np.c_[target, area_column]
                data[:, -1] = data[:, 1] * data[:, 2]  # sorted by box area
                data = data[(data[:, -1].argsort())[::-1]]  # descending order
                target = data[0:max_target_num, 0:6]

            det_1st = np.loadtxt(inputpath_1 + f)
            if len(det_1st) > 0:
                if len(det_1st.shape) == 1:
                    det_1st = det_1st[np.newaxis, :]
            if len(det_1st) > 0:
                det_1st = det_1st[np.logical_or(det_1st[:, 1] > 0.06, det_1st[:, 1] < -0.06)]
                det_1st = det_1st[np.logical_and(det_1st[:, 1] > 0, det_1st[:, 1] < 16.66)]
                det_1st = det_1st[np.logical_or(det_1st[:, 2] < -0.06, det_1st[:, 2] > 0.06)]
                det_1st = det_1st[np.logical_and(det_1st[:, 2] < 16.66, det_1st[:, 2] > -16.66)]

            det_2nd = np.loadtxt(inputpath_2 + f)
            if len(det_2nd) > 0:
                if len(det_2nd.shape) == 1:
                    det_2nd = det_2nd[np.newaxis, :]

            if len(det_2nd) > 0:
                det_2nd[:, -1] = -det_2nd[:, -1]
            #
            # | score xc yc rad |
            #

            # ang -> rads
            if len(target) > 0:
                target[:, 5] = target[:, 5] * math.pi / 180

            # filtering
            # min_xc = 0
            # max_xc = 16.66
            # if len(target) > 0:
            #     target = target[np.logical_and(target[:, 3] > min_xc, target[:, 3] < max_xc)]
            # if len(det_1st) > 0:
            #     det_1st = det_1st[
            #         np.logical_and(det_1st[:, 1] > max(min_xc - 0.45, 0), det_1st[:, 1] < min(16.66, max_xc + 0.45))]

            # min_yc = 6
            # max_yc = 16.66
            # if len(target) > 0:
            #     target = target[np.logical_and(target[:, 4] <= max_yc, target[:, 4] >= -max_yc)]
            #     target = target[np.logical_or(target[:, 4] >= min_yc, target[:, 4] <= -min_yc)]
            #
            # if len(det_1st) > 0:
            #     det_1st = det_1st[np.logical_and(det_1st[:, 2] <= max_yc, det_1st[:, 2] >= -max_yc)]
            #     det_1st = det_1st[np.logical_or(det_1st[:, 2] >= min_yc, det_1st[:, 2] <= -min_yc)]

            # min_r = 10
            # max_r = 16.66
            # if len(target) > 0:
            #     r=(target[:, 3]**2+target[:, 4]**2)**0.5
            #     target = target[np.logical_and(r > min_r, r < max_r)]
            # if len(det_1st) > 0:
            #     r=(det_1st[:, 1]**2+det_1st[:, 2]**2)**0.5
            #     det_1st = det_1st[np.logical_and(r > min_r, r < max_r)]

            # min_rad = 0.75 * math.pi
            # max_rad = 1 * math.pi
            # if len(target) > 0:
            #     rad = np.arctan2(target[:, 3], target[:, 4]) - target[:, 5]
            #     for i, d in enumerate(rad):
            #         if d > math.pi:
            #             rad[i] = d - math.pi
            #         if d > math.pi:
            #             rad[i] = d - math.pi
            #         if d < 0:
            #             rad[i] = d + math.pi
            #         if d < 0:
            #             rad[i] = d + math.pi
            #     target = target[np.logical_and(rad > min_rad, rad < max_rad)]
            #     # target = target[np.logical_or(np.logical_and(rad >= min_rad, rad <= max_rad),
            #     #                               np.logical_and(rad >= min_rad + math.pi, rad <= max_rad + math.pi))]
            # #     # print(rad)
            # if len(det_1st) > 0 and len(target) > 0:
            #     dis = 0.45
            #     D = np.zeros([len(det_1st), len(target)])
            #     for q in range(len(det_1st)):
            #         for j in range(len(target)):
            #             D[q, j] = ((det_1st[q, 1] - target[j, 3]) ** 2 + (
            #                     det_1st[q, 2] - target[j, 4]) ** 2) ** 0.5  # distance matrix
            #     det_1st = det_1st[np.min(D, axis=1) < dis]
            #
            # if len(det_1st) > 0:
            #     rad1 = np.arctan2(det_1st[:, 1], det_1st[:, 2]) - det_1st[:, 3]
            #     for i, d in enumerate(rad1):
            #         if d > math.pi:
            #             rad1[i] = d - math.pi
            #         if d > math.pi:
            #             rad1[i] = d - math.pi
            #         if d < 0:
            #             rad1[i] = d + math.pi
            #         if d < 0:
            #             rad1[i] = d + math.pi
            #
            #     # deltarad = (25 * math.pi / 180)
            #     deltarad = 0
            #     det_1st = det_1st[
            #         np.logical_and(rad1 > min_rad - deltarad, rad1 < max_rad + deltarad)]
            # det_1st = det_1st[np.logical_or(np.logical_and(rad1 >= min_rad, rad1 <= max_rad),
            #                                 np.logical_and(rad1 >= min_rad + math.pi, rad1 <= max_rad + math.pi))]

            # output fusion
            # 1 fuse 2:
            # for i, det in enumerate(det_1st):
            #     for j, det2 in enumerate(det_2nd):
            #         # a = det[1]
            #         if ((det[1] - det2[1]) ** 2 + (det[2] - det2[2]) ** 2) ** 0.5 < 0.5 and det2[0] > det[0]:
            #             # if ((det[1] - det2[1]) ** 2 + (det[2] - det2[2]) ** 2) ** 0.5 < 0.5:
            #             det_1st[i, 0] = 0.5*(det2[0]+det[0])
            #             det_1st[i, 1] = 0.5*(det2[1]+det[1])
            #             det_1st[i, 2] = 0.5*(det2[2]+det[2])
            #             det_1st[i, 3] = det[3]

            final_output = det_1st

            # thresh filtering
            if len(final_output) > 0:
                final_output = final_output[final_output[:, 0] > thresh]

            # extract values
            if len(final_output) > 0:
                outcenter, outrad = np.array([final_output[:, 1], final_output[:, 2]], dtype=float), final_output[:, 3]
            else:
                outcenter, outrad = np.array([]), np.array([])

            if len(target) > 0:
                tarcenter, tarrad = np.array([target[:, 3], target[:, 4]], dtype=float), target[:, 5]
            else:
                tarcenter, tarrad = np.array([]), np.array([])

            if len(outcenter) > 0:
                m, n = outcenter.shape
            else:
                m, n = 0, 0
            if len(tarcenter) > 0:
                o, p = tarcenter.shape
            else:
                o, p = 0, 0

            prednum += n
            tarnumraw += p
            if n == 0 or p == 0:
                continue

            # calculate in-range errors for matched pairs
            D = np.zeros([n, p])
            A = np.zeros([n, p])
            for q in range(n):
                for j in range(p):
                    D[q, j] = la.norm(outcenter[:, q] - tarcenter[:, j])  # distance matrix
                    A[q, j] = np.abs(outrad[q] - tarrad[j])
                    # A[q, j] = outrad[q] - tarrad[j]
            for ii in range(p):
                eval_distance.append(D[np.argmin(D, axis=0)[ii]][ii])
                eval_angle.append(A[np.argmin(D, axis=0)[ii]][ii])

        prerec = []
        #
        for dis in dislist:
            for ang in radlist:
                predinrange = sum(
                    # (np.array(eval_distance) < dis))  # calc matched predictions
                    (np.array(eval_distance) < dis) & ((np.array(eval_angle) < ang) | (
                            np.array(eval_angle) > (math.pi - ang))))  # calc matched predictions
                # (np.array(eval_distance) < dis) & (
                #             (np.array(eval_angle) < ang) | (np.abs(np.array(eval_angle) + math.pi) < ang)) | (
                #         np.abs(np.array(eval_angle) - math.pi) < ang))  # calc matched predictions
                # (np.array(eval_distance) < dis) & ((np.array(eval_angle) < ang) ) )  # calc matched predictions
                print(predinrange, prednum, tarnumraw, '++++', sum(np.array(eval_distance) < dis),
                      sum(np.array(eval_angle) < ang))
                pre = predinrange / prednum if prednum != 0 else 1
                rec = predinrange / tarnumraw if tarnumraw != 0 else 1
                print(' precision: %.6f' % pre, ' racall: %.6f' % rec, ' with dis', dis, 'ang', ang)
                prerec.append([pre, rec, dis, ang])

        return np.array(prerec)


def calcPR():
    thresh_list = [0, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.9, 0.99,
                   0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 1]
    # thresh_list = [0.9]
    pr_array = []

    for thresh in tqdm(thresh_list):
        prerec = detections2prerec(thresh)
        pr_array.append(prerec)
        print(thresh, '================================================')

    plot_pr_array(pr_array, dislist, radlist)


def merge_det2csv():
    dets_1st = None
    dets_2nd = None
    for fpathe, dirs, fs in os.walk(pathbox):
        for f in fs:

            # target = np.loadtxt(pathbox + f, usecols=[1, 2, 3, 5, 6, 9])  # coef, len, width, xc, yc, rad
            # if len(target) > 0:
            #     if len(target.shape) == 1:
            #         target = target[np.newaxis, :]
            # # fixing annotation mistakes
            # if len(target) > 0:
            #     target[:, 5] = -target[:, 5]
            # #
            # if len(target) > 0:
            #     target = target[np.logical_or(target[:, 3] > 0.06, target[:, 3] < -0.06)]
            #     target = target[np.logical_and(target[:, 3] > 0, target[:, 3] < 16.66)]
            #     target = target[np.logical_or(target[:, 4] < -0.06, target[:, 4] > 0.06)]
            #     target = target[np.logical_and(target[:, 4] < 16.66, target[:, 4] > -16.66)]
            #     target = target[np.logical_and(target[:, 2] > 12 / 30.0, target[:, 2] < 144 / 30.0)]
            #     target = target[np.logical_and(target[:, 1] > 12 / 30.0, target[:, 1] < 144 / 30.0)]
            #     target = target[np.logical_and(target[:, 2] * target[:, 1] > 360 / 900.0,
            #                                    target[:, 2] * target[:, 1] < 13000 / 900.0)]
            # #
            # max_target_num = 12  # 12 equals to no restriction, difference is minor
            # if len(target) > 0:
            #     area_column = np.zeros((len(target), 1), dtype=np.float)
            #     data = np.c_[target, area_column]
            #     data[:, -1] = data[:, 1] * data[:, 2]  # sorted by box area
            #     data = data[(data[:, -1].argsort())[::-1]]  # descending order
            #     target = data[0:max_target_num, 0:6]

            det_1st = np.loadtxt(inputpath_1 + f)
            if len(det_1st) > 0:
                if len(det_1st.shape) == 1:
                    det_1st = det_1st[np.newaxis, :]
                if dets_1st is None:
                    dets_1st = det_1st.copy()
                dets_1st = np.vstack((dets_1st, det_1st))

            det_2nd = np.loadtxt(inputpath_2 + f)
            if len(det_2nd) > 0:
                if len(det_2nd.shape) == 1:
                    det_2nd = det_2nd[np.newaxis, :]
                if dets_2nd is None:
                    dets_2nd = det_2nd.copy()
                dets_2nd = np.vstack((dets_2nd, det_2nd))
    a, b = dets_1st, dets_2nd
    np.savetxt('./dets_1st.csv', a, fmt='%.4f', delimiter=',', newline='\n')
    np.savetxt('./dets_2nd.csv', b, fmt='%.4f', delimiter=',', newline='\n')


def tar2csv():
    dets_tar = None
    for fpathe, dirs, fs in os.walk(pathbox):
        for f in fs:

            target = np.loadtxt(pathbox + f, usecols=[1, 2, 3, 5, 6, 9])  # coef, len, width, xc, yc, rad
            if len(target) > 0:
                if len(target.shape) == 1:
                    target = target[np.newaxis, :]
            # fixing annotation mistakes
            # if len(target) > 0:
            #     target[:, 5] = -target[:, 5]
            #
            # if len(target) > 0:
            #     target = target[np.logical_or(target[:, 3] > 0.06, target[:, 3] < -0.06)]
            #     target = target[np.logical_and(target[:, 3] > 0, target[:, 3] < 16.66)]
            #     target = target[np.logical_or(target[:, 4] < -0.06, target[:, 4] > 0.06)]
            #     target = target[np.logical_and(target[:, 4] < 16.66, target[:, 4] > -16.66)]
            #     target = target[np.logical_and(target[:, 2] > 12 / 30.0, target[:, 2] < 144 / 30.0)]
            #     target = target[np.logical_and(target[:, 1] > 12 / 30.0, target[:, 1] < 144 / 30.0)]
            #     target = target[np.logical_and(target[:, 2] * target[:, 1] > 360 / 900.0,
            #                                    target[:, 2] * target[:, 1] < 13000 / 900.0)]
            # #
            # max_target_num = 12  # 12 equals to no restriction, difference is minor
            # if len(target) > 0:
            #     area_column = np.zeros((len(target), 1), dtype=np.float)
            #     data = np.c_[target, area_column]
            #     data[:, -1] = data[:, 1] * data[:, 2]  # sorted by box area
            #     data = data[(data[:, -1].argsort())[::-1]]  # descending order
            #     target = data[0:max_target_num, 0:6]

            if len(target) > 0:
                target[:, 5] = target[:, 5] * math.pi / 180

            if len(target) > 0:
                if dets_tar is None:
                    dets_tar = target.copy()
                dets_tar = np.vstack((dets_tar, target))
    np.savetxt('./dets_tar_ori.csv', dets_tar, fmt='%.4f', delimiter=',', newline='\n')


if __name__ == '__main__':
    calcPR()
    # merge_det2csv(
