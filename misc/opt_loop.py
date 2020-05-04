# -*- coding: utf-8 -*-

"""
ガウスザイデル法を用いたループ計算
Author :
    Yuki Kumon
Last Update :
    2020-05-04
"""


import numpy as np


# ガウズ=ザイデル法の計算ループ、横方向(バイラテラルフィルターバージョン)
def optimize_loop_bilateral_horizon(img_dis, color_weight_matrix, gausian_weight, coefficient, alpha, exclusion, size):
    # w = exclusion * 2 + 1
    color_weight = np.zeros_like(gausian_weight)
    sub_img = np.zeros_like(gausian_weight)

    error = 0.0

    for i in range(exclusion, size[0] - exclusion - 1):
        for j in range(exclusion, size[1] - exclusion - 1):
            sub_img = img_dis[i - exclusion: i + exclusion + 1, j - exclusion: j + exclusion + 1]
            color_weight = color_weight_matrix[i - exclusion, j - exclusion]
            # 式中のaとbを計算する
            a = -(coefficient[exclusion, exclusion] - (coefficient[exclusion, exclusion + 1] + coefficient[exclusion, exclusion - 1]) / 2.0)
            b = sub_img[exclusion, exclusion] - (coefficient[exclusion, exclusion + 1] - coefficient[exclusion, exclusion - 1]) / 2.0 / (-2.0 * coefficient[exclusion, exclusion] + coefficient[exclusion, exclusion + 1] + coefficient[exclusion, exclusion - 1])
            # このとき、gausian_weight * color_weightがフィルターの重みαijに相当する
            # d_new = (-coefficient[i, j] * sub_img[exclusion, exclusion] + (gausian_weight * color_weight * sub_img).sum()) / (-coefficient[i, j] + (gausian_weight * color_weight).sum())
            d_new = (-a * b + (gausian_weight * color_weight * sub_img).sum()) / (-a + (gausian_weight * color_weight).sum())
            # この評価指標は暫定的なもの
            error += abs(img_dis[i, j] - d_new)
            img_dis[i, j] = d_new
    return img_dis, error


# ガウズ=ザイデル法の計算ループ、縦方向(バイラテラルフィルターバージョン)
def optimize_loop_bilateral_vertical(img_dis, color_weight_matrix, gausian_weight, coefficient, alpha, exclusion, size):
    # w = exclusion * 2 + 1
    color_weight = np.zeros_like(gausian_weight)
    sub_img = np.zeros_like(gausian_weight)

    error = 0.0

    for i in range(exclusion, size[0] - exclusion - 1):
        for j in range(exclusion, size[1] - exclusion - 1):
            sub_img = img_dis[i - exclusion: i + exclusion + 1, j - exclusion: j + exclusion + 1]
            color_weight = color_weight_matrix[i - exclusion, j - exclusion]
            # 式中のaとbを計算する
            a = -(coefficient[exclusion, exclusion] - (coefficient[exclusion + 1, exclusion] + coefficient[exclusion - 1, exclusion]) / 2.0)
            b = sub_img[exclusion, exclusion] - (coefficient[exclusion + 1, exclusion] - coefficient[exclusion - 1, exclusion]) / 2.0 / (-2.0 * coefficient[exclusion, exclusion] + coefficient[exclusion + 1, exclusion] + coefficient[exclusion - 1, exclusion])
            # このとき、gausian_weight * color_weightがフィルターの重みαijに相当する
            # d_new = (-coefficient[i, j] * sub_img[exclusion, exclusion] + (gausian_weight * color_weight * sub_img).sum()) / (-coefficient[i, j] + (gausian_weight * color_weight).sum())
            d_new = (-a * b + (gausian_weight * color_weight * sub_img).sum()) / (-a + (gausian_weight * color_weight).sum())
            # この評価指標は暫定的なもの
            error += abs(img_dis[i, j] - d_new)
            img_dis[i, j] = d_new
    return img_dis, error


# 重み付け配列を作成
def make_weight(guide_img, exclusion, size, sigma):
    w = exclusion * 2 + 1
    gausian_weight = np.zeros([w, w])
    color_weight = np.zeros_like(gausian_weight)
    sub_guide = np.zeros_like(gausian_weight)
    color_weight_matrix = np.zeros([size[0] - exclusion, size[1] - exclusion, w, w])

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
