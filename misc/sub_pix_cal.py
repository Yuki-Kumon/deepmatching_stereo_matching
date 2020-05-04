# -*- coding: utf-8 -*-

"""
sub pix calculatioon
Author :
    Yuki Kumon
Last Update :
    2020-01-14
"""


import os
import sys

import numpy as np

sys.path.append('.')

from misc.optimize_loop import image_threshold


def sub_pix_cal(arr, co_map, direction=0, ratio=100.):
    arr = image_threshold(arr, threshold=[-3, 3])
    arr = arr.astype(float)

    def index_plus(lis):
        return [int(lis[i] + 1) if i == direction else int(lis[i]) for i in range(2)]

    def index_minus(lis):
        return [int(lis[i] - 1) if i == direction else int(lis[i]) for i in range(2)]

    for i in range(1, arr.shape[0] - 1):
        for j in range(1, arr.shape[1] - 1):
            try:
                # 二次関数近似のため近傍の相関値を取得
                plus = index_plus([i, j])
                minus = index_minus([i, j])
                d = arr[i, j]
                r0 = co_map[i, j] * ratio
                r1 = co_map[plus[0], plus[1]] * ratio
                r_ = co_map[minus[0], minus[1]] * ratio
                dis = d - (r1 - r_) / (2 * (r1 + r_ - 2 * r0))
                if abs(d - dis) > 1:
                    dis = d
            except IndexError as e:
                dis = d
                # exception_count += 1
            # update
            arr[i, j] = dis
    arr = image_threshold(arr, threshold=[-3, 3])
    # arr = image_threshold(arr, threshold=[-3, 3])

    return arr


if __name__ == '__main__':
    from misc.image_cut_solver import *
    if 1:
        res1 = sub_pix_cal(
            np.load('./output/igarss/setup1/elevation.npy'),
            np.load('./output/igarss/setup1/correlation.npy'),
            direction=0
        )
        ImageCutSolver.image_save('./output/igarss/setup1/elevation_sub_pix.png', res1 * 90 + 100, threshold=[10, 250])
    if 1:
        res2 = sub_pix_cal(
            np.load('./output/igarss/setup1/elevation2.npy'),
            np.load('./output/igarss/setup1/correlation.npy'),
            direction=0
        )
        ImageCutSolver.image_save('./output/igarss/setup1/elevation2_sub_pix.png', res2 * 90 + 100, threshold=[10, 250])
    # print(np.load('./output/igarss/setup1/elevation2.npy')[200, 500])
