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


# フィルターをかけるために、元画像の大きさを拡張する(バイラテラルフィルター用)
cdef extension(np.ndarray[DOUBLE_t, ndim=2] img, int W, int H, int d):
    # ループ用の変数
    cdef int i, j
    # 結果を出力するための変数
    cdef np.ndarray[DOUBLE_t, ndim=2] result = np.zeros([W + (d - 1), H + (d - 1)])

    # 周以外の点を挿入
    result[(d - 1) / 2:-(d - 1) / 2, (d - 1) / 2:-(d - 1) / 2] = img

    # 外周に点を挿入する(四隅以外)
    for i in range((d - 1) / 2):
        result[i, (d - 1) / 2:-(d - 1) / 2] = img[0, :]
        result[-i - 1, (d - 1) / 2:-(d - 1) / 2] = img[-1, :]
    for j in range((d - 1) / 2):
        result[(d - 1) / 2:-(d - 1) / 2, j] = img[:, 0]
        result[(d - 1) / 2:-(d - 1) / 2, -j - 1] = img[:, -1]
    # 四隅の点を挿入する
    for i in range((d - 1) / 2):
        for j in range((d - 1) / 2):
            result[i, j] = img[0, 0]
            result[i, -j - 1] = img[0, -1]
            result[-i - 1, j] = img[-1, 0]
            result[-i - 1, -j - 1] = img[-1, -1]
    return result


# ガウズ=ザイデル法の計算ループ(バイラテラルフィルターバージョン)
def optimize_loop_bilateral_old(np.ndarray[DOUBLE_t, ndim=2] img_dis, np.ndarray[DOUBLE_t, ndim=2] coefficient, float alpha, int exclusion, np.ndarray[INT_t, ndim=1] size, np.ndarray[DOUBLE_t, ndim=1] sigma):
    # ループ用変数
    cdef int i, j, w
    w = exclusion * 2 + 1
    # 計算用変数
    cdef float error, d_new
    cdef np.ndarray[DOUBLE_t, ndim=2] gausian_weight = np.zeros([w, w])
    cdef np.ndarray[DOUBLE_t, ndim=2] color_weight = np.zeros_like(gausian_weight)
    cdef np.ndarray[DOUBLE_t, ndim=2] sub_img = np.zeros_like(gausian_weight)
    # cdef np.ndarray[DOUBLE_t, ndim=2] img_dis2 = np.zeros([size[0] + w, size[1] + w])

    # 画像の拡張
    # img_dis2 = extension(img_dis, size[0], size[1], w)

    # 空間用の重み付け配列の準備
    # 空間の重み付け用の配列を準備しておく
    for i in range(w):
        for j in range(w):
            gausian_weight[i, j] = np.exp(-(float((i - exclusion)**2 + (j - exclusion)**2)) / (2.0 * sigma[1]**2))

    for i in range(exclusion, size[0] - exclusion - 1):
        for j in range(exclusion, size[1] - exclusion - 1):
            sub_img = img_dis[i - exclusion: i + exclusion + 1, j - exclusion: j + exclusion + 1]
            # 輝度の重みつけ配列の計算
            color_weight = sub_img[exclusion, exclusion] - sub_img
            color_weight = -1 * color_weight * color_weight / (2.0 * sigma[0]**2)
            color_weight = np.exp(color_weight)
            # このとき、gausian_weight * color_weightがフィルターの重みαijに相当する
            d_new = (-coefficient[i, j] * sub_img[exclusion, exclusion] + (gausian_weight * color_weight * sub_img).sum()) / (-coefficient[i, j] + (gausian_weight * color_weight).sum())
            error += abs(img_dis[i, j] - d_new)
            # print(str(img_dis[i, j]) + ',' + str(sum_d) + ',' + str(d_new) + ',' + str(coefficient[i, j]) + ',' + str(error))
            img_dis[i, j] = d_new

    """
    # ループ計算(画像の拡張の影響でインデックスがずれている。注意！！！)
    error = 0.0
    for i in range(exclusion, size[0] + exclusion):
        for j in range(exclusion, size[1] + exclusion):
            sub_img = img_dis2[i - exclusion: i - exclusion + w, j - exclusion: j - exclusion + w]
            # 輝度の重みつけ配列の計算
            color_weight = sub_img[exclusion, exclusion] - sub_img
            color_weight = -1 * color_weight * color_weight / (2.0 * sigma[0]**2)
            color_weight = np.exp(color_weight)
            # このとき、gausian_weight * color_weightがフィルターの重みαijに相当する
            # ijが0の時を除く必要？(ガウシアンの重みの方に実装)←これすると重みの和がバグるのでマズイ！かもしれない...
            d_new = (-coefficient[i - exclusion, j - exclusion] * sub_img[exclusion, exclusion] + (gausian_weight * color_weight * sub_img).sum()) / (-coefficient[i - exclusion, j - exclusion] + (gausian_weight * color_weight).sum())
            error += abs(img_dis2[i, j] - d_new)
            # print(str(img_dis[i, j]) + ',' + str(sum_d) + ',' + str(d_new) + ',' + str(coefficient[i, j]) + ',' + str(error))
            img_dis2[i, j] = d_new
    img_dis = img_dis2[exclusion: -exclusion, exclusion: -exclusion]
    """

    return img_dis, error


# ガウズ=ザイデル法の計算ループ(バイラテラルフィルターバージョン)
def optimize_loop_bilateral_old2(np.ndarray[DOUBLE_t, ndim=2] img_dis, np.ndarray[DOUBLE_t, ndim=2] guide_img, np.ndarray[DOUBLE_t, ndim=2] coefficient, float alpha, int exclusion, np.ndarray[INT_t, ndim=1] size, np.ndarray[DOUBLE_t, ndim=1] sigma):
    # ループ用変数
    cdef int i, j, w
    w = exclusion * 2 + 1
    # 計算用変数
    cdef float error, d_new
    cdef np.ndarray[DOUBLE_t, ndim=2] gausian_weight = np.zeros([w, w])
    cdef np.ndarray[DOUBLE_t, ndim=2] color_weight = np.zeros_like(gausian_weight)
    cdef np.ndarray[DOUBLE_t, ndim=2] sub_img = np.zeros_like(gausian_weight)

    # 空間用の重み付け配列の準備
    # 空間の重み付け用の配列を準備しておく
    for i in range(w):
        for j in range(w):
            gausian_weight[i, j] = np.exp(-(float((i - exclusion)**2 + (j - exclusion)**2)) / (2.0 * sigma[1]**2))

    for i in range(exclusion, size[0] - exclusion - 1):
        for j in range(exclusion, size[1] - exclusion - 1):
            sub_img = img_dis[i - exclusion: i + exclusion + 1, j - exclusion: j + exclusion + 1]
            sub_guide = guide_img[i - exclusion: i + exclusion + 1, j - exclusion: j + exclusion + 1]
            # 輝度の重みつけ配列の計算
            color_weight = sub_guide[exclusion, exclusion] - sub_guide
            color_weight = -1 * color_weight * color_weight / (2.0 * sigma[0]**2)
            color_weight = np.exp(color_weight)
            # このとき、gausian_weight * color_weightがフィルターの重みαijに相当する
            d_new = (-coefficient[i, j] * sub_img[exclusion, exclusion] + (gausian_weight * color_weight * sub_img).sum()) / (-coefficient[i, j] + (gausian_weight * color_weight).sum())
            error += abs(img_dis[i, j] - d_new)
            # print(str(img_dis[i, j]) + ',' + str(sum_d) + ',' + str(d_new) + ',' + str(coefficient[i, j]) + ',' + str(error))
            img_dis[i, j] = d_new
    return img_dis, error

# ガウズ=ザイデル法の計算ループ(バイラテラルフィルターバージョン)
def optimize_loop_bilateral_old2(np.ndarray[DOUBLE_t, ndim=2] img_dis, np.ndarray[DOUBLE_t, ndim=2] guide_img, np.ndarray[DOUBLE_t, ndim=2] coefficient, float alpha, int exclusion, np.ndarray[INT_t, ndim=1] size, np.ndarray[DOUBLE_t, ndim=1] sigma):
    # ループ用変数
    cdef int i, j, w
    w = exclusion * 2 + 1
    # 計算用変数
    cdef float error, d_new
    cdef np.ndarray[DOUBLE_t, ndim=2] gausian_weight = np.zeros([w, w])
    cdef np.ndarray[DOUBLE_t, ndim=2] color_weight = np.zeros_like(gausian_weight)
    cdef np.ndarray[DOUBLE_t, ndim=2] sub_img = np.zeros_like(gausian_weight)

    # 空間用の重み付け配列の準備
    # 空間の重み付け用の配列を準備しておく
    for i in range(w):
        for j in range(w):
            gausian_weight[i, j] = np.exp(-(float((i - exclusion)**2 + (j - exclusion)**2)) / (2.0 * sigma[1]**2))

    for i in range(exclusion, size[0] - exclusion - 1):
        for j in range(exclusion, size[1] - exclusion - 1):
            sub_img = img_dis[i - exclusion: i + exclusion + 1, j - exclusion: j + exclusion + 1]
            sub_guide = guide_img[i - exclusion: i + exclusion + 1, j - exclusion: j + exclusion + 1]
            # 輝度の重みつけ配列の計算
            color_weight = sub_guide[exclusion, exclusion] - sub_guide
            color_weight = -1 * color_weight * color_weight / (2.0 * sigma[0]**2)
            color_weight = np.exp(color_weight)
            # このとき、gausian_weight * color_weightがフィルターの重みαijに相当する
            d_new = (-coefficient[i, j] * sub_img[exclusion, exclusion] + (gausian_weight * color_weight * sub_img).sum()) / (-coefficient[i, j] + (gausian_weight * color_weight).sum())
            error += abs(img_dis[i, j] - d_new)
            # print(str(img_dis[i, j]) + ',' + str(sum_d) + ',' + str(d_new) + ',' + str(coefficient[i, j]) + ',' + str(error))
            img_dis[i, j] = d_new
    return img_dis, error
