# -*- coding: utf-8 -*-

"""
correlation map version 2
Author :
    Yuki Kumon
Last Update :
    2019-12-17
"""


import sys
sys.path.append('.')

import numpy as np
import cv2

import torch
import torch.nn as nn

from joblib import Parallel, delayed

from misc.Feature_value import Feature_value


class CorrelationMapV2():
    '''
    deepmatchingみたいにピラミッド状の特徴マップを作成する
    '''

    def __init__(self, img, template, window_size=3, deep_level=4, r_lam=1.4, feature_name='cv2.TM_CCOEFF_NORMED'):
        '''
        window_sizeは奇数
        Nは自然数
        '''
        N = deep_level
        assert img.shape == template.shape, '2枚の画像は同じサイズのみ'
        self.img = img
        self.template = template
        self.window_size = window_size
        self.max_N = N
        self.lam = r_lam

        self.exclusive_pix = int((window_size) / 2)

        assert (img.shape[0] - self.exclusive_pix * 2) % (2**N) == 0, '画像の横の長さが割り切れない'
        assert (img.shape[1] - self.exclusive_pix * 2) % (2**N) == 0, '画像の縦の長さが割り切れない'
        self.image_size = [x for x in img.shape]
        # ピラミッド頂上でのマップサイズを取得
        self.i_global_max = int(img.shape[0] - self.exclusive_pix * 2) / (2**N)
        self.j_global_max = int(img.shape[1] - self.exclusive_pix * 2) / (2**N)

        self.Feature = Feature_value(feature_name=feature_name)
        self.Maxpool = Maxpool()
        self.Maxpool.eval()

    def _create_atomic_patch(self):
        '''
        重なりありのatomic patchを作成する
        '''
        atomic_patch = np.empty((
                                self.image_size[0] - 2 * self.exclusive_pix,
                                self.image_size[1] - 2 * self.exclusive_pix,
                                self.window_size,
                                self.window_size
                                ))
        for i in range(self.exclusive_pix, self.image_size[0] - self.exclusive_pix):
            for j in range(self.exclusive_pix, self.image_size[1] - self.exclusive_pix):
                atomic_patch[i - self.exclusive_pix, j - self.exclusive_pix] = \
                    self.img[i - self.exclusive_pix:i + self.exclusive_pix + 1, j - self.exclusive_pix:j + self.exclusive_pix + 1]

        # データ型を符号なし整数に変えておく
        self.atomic_patch = atomic_patch.astype(np.uint8)

    def _create_initial_co_map(self):
        '''
        初めの相関マップを計算する。
        co_map[i, j] = [2次元相関マップ(Nに基づく近傍のみ計算)]
        '''
        co_map = np.empty((
            self.atomic_patch.shape[0],
            self.atomic_patch.shape[1],
            2**self.max_N,
            2**self.max_N
        ))
        for i in range(self.exclusive_pix, self.image_size[0] - self.exclusive_pix):
            for j in range(self.exclusive_pix, self.image_size[1] - self.exclusive_pix):
                # 座標取得
                i_list, j_list = self._template_index(i, j)
                co_here = self.Feature(
                    self.atomic_patch[i - self.exclusive_pix, j - self.exclusive_pix],
                    self.template[i_list[0]:i_list[1], j_list[0]:j_list[1]]
                )
                co_map[i - self.exclusive_pix, j - self.exclusive_pix] = co_here
        self.co_map = co_map

    def _aggregation(self, map):
        '''
        aggregation to make upper class co_map
        pooling window:3
        stride:2
        パディングして解像度が半分になることが大切
        '''
        map_len_1 = int((map.shape[2]) / 2)
        map_len_2 = int((map.shape[3]) / 2)
        res = np.empty((map.shape[0], map.shape[1], map_len_1, map_len_2))

        # max pool
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                res[i, j] = self.Maxpool(torch.from_numpy(map[i, j][None])).numpy()[0]

        # shift and average
        output = np.empty((map_len_1, map_len_2, map_len_1, map_len_2))

        # process
        def process(i, j, res):
            # 平均を取る対象のインデックスを計算しておく
            upper_left = [i * 2, j * 2]
            upper_right = [i * 2, j * 2 + 1]
            lower_left = [i * 2 + 1, j * 2]
            lower_right = [i * 2 + 1, j * 2 + 1]

            # shiftはpaddingしているので不要
            upper_left_img = res[upper_left[0], upper_left[1]]
            upper_right_img = res[upper_right[0], upper_right[1]]
            lower_left_img = res[lower_left[0], lower_left[1]]
            lower_right_img = res[lower_right[0], lower_right[1]]
            # average
            return (upper_left_img + upper_right_img + lower_left_img + lower_right_img) / 4, (i, j)

        # parallel processing
        response = Parallel(n_jobs=-1)([delayed(process)(i, j, res) for i in range(map_len_1) for j in range(map_len_2)])
        # assign
        for res_here in response:
            output[res_here[1]] = res_here[0]

        return output

    def _multi_level_correlation_pyramid(self):
        '''
        aggregationを繰り返し、multi-level correlation pyramidを計算する
        初めの特徴マップはFeatureに依存(現時点では原著と異なりznccを使っている)
        '''

        # 原著の手順に従ってmulti-level correlation pyramidを計算する
        co_map_list = []
        co_map = self.co_map
        co_map = self._rectification(co_map)
        co_map_list.append(co_map)
        N = 1
        iteration = 1
        for N_here in range(self.max_N + 1):
            # aggregation
            aggregated_map = self._aggregation(co_map)
            aggregated_map = self._rectification(aggregated_map)
            co_map_list.append(aggregated_map)
            del co_map
            co_map = aggregated_map
            N *= 2
            iteration += 1
        self.co_map_list = co_map_list
        self.iteration = iteration
        self.N_map = N

    def _rectification(self, map):
        return map**self.lam

    def _template_index(self, i, j):
        """
        最大のNの値に応じて画像の切り方が変わる
        それを返す
        """
        # 除外ピクセル分調整
        i = i - self.exclusive_pix
        j = j - self.exclusive_pix
        # ピラミッド頂上のパッチ内イテレーション
        i_in = i % (2**self.max_N)
        j_in = j % (2**self.max_N)
        # ピラミッド頂上でのイテレーション
        i_global = int((i - i_in) / (2**self.max_N))
        j_global = int((j - j_in) / (2**self.max_N))

        # テンプレート画像の対応座標
        i_list = [2**self.max_N * i_global, 2**self.max_N * (i_global + 1) + 2 * self.exclusive_pix]
        j_list = [0, 2**self.max_N + 2 * self.exclusive_pix]

        return i_list, j_list




class Maxpool(nn.Module):

    def __init__(self, window=3, stride=2, padding=1):
        super(Maxpool, self).__init__()

        self.pool = nn.MaxPool2d(window, stride, padding=padding)

    def forward(self, x):
        return self.pool(x)


if __name__ == '__main__':
    """
    sanity check
    """
    # atomicな特徴マップが一辺が2^nじゃないとバグるカス実装です。。。
    img1 = cv2.imread('./data/band3s.tif', cv2.IMREAD_GRAYSCALE)[:130, :130]
    img2 = cv2.imread('./data/band3bs.tif', cv2.IMREAD_GRAYSCALE)[:130, :130]

    cls = CorrelationMapV2(img1, img2)
    cls._create_atomic_patch()
    cls._create_initial_co_map()
    # cls._aggregation(cls.co_map)
    cls._multi_level_correlation_pyramid()
