# -*- coding: utf-8 -*-

"""
correlation map version 2
Author :
    Yuki Kumon
Last Update :
    2019-12-05
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

    def __init__(self, img, template, window_size=3, search_window_size=[9, 9], r_lam=1.4, feature_name='cv2.TM_CCOEFF_NORMED'):
        '''
        各window_sizeは奇数
        search_window_sizeはリストで指定(今の所正方形のみ)
        '''
        if img.shape != template.shape:
            print('use same size images!(リサイズすると意味がないので)')
            sys.exit()
        self.img = img
        self.template = template
        self.window_size = window_size
        self.search_window_size = search_window_size
        self.lam = r_lam

        self.exclusive_pix = int((window_size - 1) / 2)
        # cv2 matchi templateではパディングを行わないのでここで大きめに取っておく
        self.search_range = [int((search_window_size[i] - 1) / 2) + self.exclusive_pix for i in range(2)]
        self.template_padding_pix = [int((search_window_size[i] - 1) / 2) for i in range(2)]
        self.image_size = [x for x in img.shape]

        self.Feature = Feature_value(feature_name=feature_name)
        self.Maxpool = Maxpool()
        self.Maxpool.eval()
        self.Maxpool_p = Maxpool_padding()
        self.Maxpool_p.eval()

    def _create_atomic_patch(self):
        '''
        atomic patchの作成
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
        N=1の相関マップの計算
        ただし、近傍のみで計算を行う
        '''
        co_map = np.empty((
            self.atomic_patch.shape[0],
            self.atomic_patch.shape[1],
            self.search_window_size[0],
            self.search_window_size[1]
        ))
        for i in range(self.exclusive_pix, self.image_size[0] - self.exclusive_pix):
            for j in range(self.exclusive_pix, self.image_size[1] - self.exclusive_pix):
                # 各画素ごとにループ計算を実施する
                co_here = self.Feature(
                    self.atomic_patch[i - self.exclusive_pix, j - self.exclusive_pix],  # 元画像
                    self.template[
                        self.template_padding_pix[0] + i - self.search_range[0]:
                        self.template_padding_pix[0] + i + self.search_range[0] + 1,
                        self.template_padding_pix[1] + j - self.search_range[1]:
                        self.template_padding_pix[1] + j + self.search_range[1] + 1
                    ]  # テンプレート画像(近傍)
                )
                co_map[i - self.exclusive_pix, j - self.exclusive_pix] = co_here
        self.co_map = co_map

    def _aggregation(self, map):
        '''
        特徴マップを圧縮し荒いマップを計算していく
        input: (W, H, w, h)
        output: (W/2, H/2, w/2, h/2)
        '''
        map_len_1 = int((map.shape[2]) / 2)
        map_len_2 = int((map.shape[3]) / 2)
        map_im_len_1 = int((map.shape[0]) / 2)
        map_im_len_2 = int((map.shape[1]) / 2)
        # 特徴マップは半分の解像度になる(maxpoolingの設定を変えれば変わる)
        res = np.empty((map.shape[0], map.shape[1], map_len_1, map_len_2))

        # max pool
        if (map_len_1 % 2 == 0):
            for i in range(map.shape[0]):
                for j in range(map.shape[1]):
                    # 各画素に対応する特徴マップに対してmax poolingする
                    res[i, j] = self.Maxpool(torch.from_numpy(map[i, j][None])).numpy()[0]
        else:
            for i in range(map.shape[0]):
                for j in range(map.shape[1]):
                    # 各画素に対応する特徴マップに対してmax poolingする
                    res[i, j] = self.Maxpool_p(torch.from_numpy(map[i, j][None])).numpy()[0]

        # shift and average
        output = np.empty((map_im_len_1, map_im_len_2, map_len_1, map_len_2))
        # print(res[-1, -1])

        # process
        def process(i, j, res):
            # 平均を取る対象のインデックスを計算しておく
            # 各画素ごとに画像全体に対するインデックスとのズレに注意する
            upper_left = [i * 2, j * 2]
            upper_right = [i * 2, j * 2 + 1]
            lower_left = [i * 2 + 1, j * 2]
            lower_right = [i * 2 + 1, j * 2 + 1]

            # shiftは各画素ごとに特徴マップの元画像での位置がずれているため不要
            upper_left_img = res[upper_left[0], upper_left[1]]
            upper_right_img = res[upper_right[0], upper_right[1]]
            lower_left_img = res[lower_left[0], lower_left[1]]
            lower_right_img = res[lower_right[0], lower_right[1]]
            # average
            return (upper_left_img + upper_right_img + lower_left_img + lower_right_img) / 4, (i, j)

        # parallel processing
        response = Parallel(n_jobs=-1)([delayed(process)(i, j, res) for i in range(map_im_len_1) for j in range(map_im_len_2)])
        # assign
        for res_here in response:
            output[res_here[1]] = res_here[0]

        return output

    def _template_padding(self):
        '''
        相関マップの計算のためにゼロパディングする
        '''
        temp = self.template
        self.template = np.zeros([
            self.img.shape[0] + 2 * self.template_padding_pix[0],
            self.img.shape[1] + 2 * self.template_padding_pix[1]
        ])
        self.template[
            self.template_padding_pix[0]:-self.template_padding_pix[0],
            self.template_padding_pix[0]:-self.template_padding_pix[0]
        ] = temp
        self.template = self.template.astype(np.uint8)


class Maxpool_padding(nn.Module):

    def __init__(self, window=3, stride=2, padding=1):
        super(Maxpool_padding, self).__init__()

        self.pool = nn.MaxPool2d(window, stride, padding=padding)

    def forward(self, x):
        return self.pool(x)


class Maxpool(nn.Module):

    def __init__(self, window=3, stride=2, padding=0):
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

    cls._template_padding()
    cls._create_atomic_patch()
    cls._create_initial_co_map()
    print(cls.co_map.shape)
    out = cls._aggregation(cls.co_map)
    print(out.shape)
    out = cls._aggregation(out)
    print(out.shape)
    out = cls._aggregation(out)
    print(out)
