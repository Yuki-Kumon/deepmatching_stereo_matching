# -*- coding: utf-8 -*-

"""
ガウス=ザイデル法のループ計算です。
Usage :
    $ python setup.py build_ext --inplace
Author :
    Yuki Kumon
Last Update :
    2018-11-19
"""


# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE_t
ctypedef np.int_t INT_t


# ガウズ=ザイデル法の計算ループ
def optimize_loop(np.ndarray[DOUBLE_t, ndim=2] img_dis, np.ndarray[DOUBLE_t, ndim=2] coefficient, float alpha, int exclusion, np.ndarray[INT_t, ndim=1]  size):
    # ループ用変数
    cdef int i, j
    # 計算用変数
    cdef float error, a, d_new
    error = 0.0
    for i in range(exclusion, size[0] - exclusion - 1):
        for j in range(exclusion, size[1] - exclusion - 1):
            sum_d = img_dis[i, j - 1] + img_dis[i, j + 1] + img_dis[i - 1, j] + img_dis[i + 1, j]
            # a = coefficient[i, j] * (check[i, j] + 1.0)
            a = coefficient[i, j]
            d_new = (-a * img_dis[i, j] + alpha * sum_d) / (-a + 4.0 * alpha)
            img_dis[i, j] = d_new
    # 逆から
    for i in range(exclusion, size[0] - exclusion - 1):
        for j in range(exclusion, size[1] - exclusion - 1):
            i = size[0] - i - 1
            j = size[1] - j - 1
            sum_d = img_dis[i, j - 1] + img_dis[i, j + 1] + img_dis[i - 1, j] + img_dis[i + 1, j]
            a = coefficient[i, j]
            d_new = (-a * img_dis[i, j] + alpha * sum_d) / (-a + 4.0 * alpha)
            error += abs(img_dis[i, j] - d_new)
            # print(str(img_dis[i, j]) + ',' + str(sum_d) + ',' + str(d_new) + ',' + str(coefficient[i, j]) + ',' + str(error))
            img_dis[i, j] = d_new
    return img_dis, error


# ガウズ=ザイデル法の計算ループ(バイラテラルフィルターバージョン)
def optimize_loop_bilateral(np.ndarray[DOUBLE_t, ndim=2] img_dis, np.ndarray[DOUBLE_t, ndim=4] color_weight_matrix, np.ndarray[DOUBLE_t, ndim=2] gausian_weight, np.ndarray[DOUBLE_t, ndim=2] coefficient, float alpha, int exclusion, np.ndarray[INT_t, ndim=1] size):
    # ループ用変数
    cdef int i, j, w
    w = exclusion * 2 + 1
    # 計算用変数
    cdef float error, d_new
    cdef np.ndarray[DOUBLE_t, ndim=2] color_weight = np.zeros_like(gausian_weight)
    cdef np.ndarray[DOUBLE_t, ndim=2] sub_img = np.zeros_like(gausian_weight)


    for i in range(exclusion, size[0] - exclusion - 1):
        for j in range(exclusion, size[1] - exclusion - 1):
            sub_img = img_dis[i - exclusion: i + exclusion + 1, j - exclusion: j + exclusion + 1]
            color_weight = color_weight_matrix[i - exclusion, j - exclusion]
            # このとき、gausian_weight * color_weightがフィルターの重みαijに相当する
            d_new = (-coefficient[i, j] * sub_img[exclusion, exclusion] + (gausian_weight * color_weight * sub_img).sum()) / (-coefficient[i, j] + (gausian_weight * color_weight).sum())
            # この評価指標は暫定的なもの
            error += abs(img_dis[i, j] - d_new)
            img_dis[i, j] = d_new
    return img_dis, error


# 重み付け配列を作成
def make_weight(np.ndarray[DOUBLE_t, ndim=2] guide_img, int exclusion, np.ndarray[INT_t, ndim=1] size, np.ndarray[DOUBLE_t, ndim=1] sigma):
    # ループ用変数
    cdef int i, j, w
    w = exclusion * 2 + 1
    # 計算用変数
    cdef float error, d_new
    cdef np.ndarray[DOUBLE_t, ndim=2] gausian_weight = np.zeros([w, w])
    cdef np.ndarray[DOUBLE_t, ndim=2] color_weight = np.zeros_like(gausian_weight)
    cdef np.ndarray[DOUBLE_t, ndim=2] sub_guide = np.zeros_like(gausian_weight)
    cdef np.ndarray[DOUBLE_t, ndim=4] color_weight_matrix = np.zeros([size[0] - exclusion, size[1] - exclusion, w, w])

    # 空間の重み付け用の配列を準備しておく
    for i in range(w):
        for j in range(w):
            gausian_weight[i, j] = np.exp(-(float((i - exclusion)**2 + (j - exclusion)**2)) / (2.0 * sigma[1]**2))

    # 輝度の重み付け配列
    for i in range(exclusion, size[0] - exclusion - 1):
        for j in range(exclusion, size[1] - exclusion - 1):
            sub_guide = guide_img[i - exclusion: i + exclusion + 1, j - exclusion: j + exclusion + 1]
            # 輝度の重みつけ配列の計算
            color_weight = sub_guide[exclusion, exclusion] - sub_guide
            color_weight = -1 * color_weight * color_weight / (2.0 * sigma[0]**2)
            color_weight = np.exp(color_weight)
            color_weight_matrix[i - exclusion, j - exclusion] = color_weight
    return gausian_weight, color_weight_matrix


# 事後確率を計算する
cdef calc_posterior(float d, float d_new, float average):
    # likelihoodはdを中心とした分散sigma1の正規分布
    # priorはバイラテラルフィルターで決まる視差を中心とした正規分布
    a = 0
