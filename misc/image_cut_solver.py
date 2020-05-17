# -*- coding: utf-8 -*-

"""
image cut tool
Author :
    Yuki Kumon
Last Update :
    2019-12-22
"""


import numpy as np
from tqdm import trange

from PIL import Image

import sys
sys.path.append('.')

from misc.Correlation_map import Correlation_map
from misc.Matching import Matching
from misc.Calc_difference import Calc_difference
# from misc.sub_pix_cal import *


class ImageCutSolver():
    '''
    小画像に切ってそれぞれdeepmatchingに入れる
    '''

    def __init__(
        self, img1, img2,
        image_size=[32, 32], stride=[32, 32], window_size=5,
        feature_name='cv2.TM_CCOEFF_NORMED', degree_map_mode=['elevation'],
        padding=False,
        sub_pix=True,
        filtering=False,
        filtering_window_size=3,
        filtering_num=3
    ):
        """
        image_sizeとstride: pyramidの深さに寄与
        window_size: atomic pathの大きさに寄与
        """
        self.img_shape = img1.shape
        assert self.img_shape == img2.shape, '2枚の画像は同じサイズ！'
        self.img1 = img1
        self.img2 = img2
        self.stride = stride
        self.window_size = window_size
        self.degree_map_mode = degree_map_mode
        self.exclusive_pix = int((window_size - 1) / 2)
        self.image_size = image_size
        self.trimed_size = [image_size[i] + 2 * self.exclusive_pix for i in range(2)]
        self.feature_name = feature_name

        if padding:
            self._padding()

        # loop length
        self.len = [int(np.floor((self.img_shape[i] - self.trimed_size[i]) / self.stride[i])) for i in range(2)]

        self.padding = padding
        self.sub_pix = sub_pix
        self.filtering = filtering
        self.filtering_window_size = filtering_window_size
        self.filtering_num = filtering_num

        self.log_flg = True

    def _padding(self):
        img1 = self.img1
        img2 = self.img2
        self.img1 = np.zeros([
            img1.shape[0] + 2 * self.exclusive_pix,
            img1.shape[1] + 2 * self.exclusive_pix
        ])
        self.img1[
            self.exclusive_pix:-self.exclusive_pix,
            self.exclusive_pix:-self.exclusive_pix
        ] = img1
        self.img1 = self.img1.astype(np.uint8)
        self.img2 = np.zeros([
            img2.shape[0] + 2 * self.exclusive_pix,
            img2.shape[1] + 2 * self.exclusive_pix
        ])
        self.img1[
            self.exclusive_pix:-self.exclusive_pix,
            self.exclusive_pix:-self.exclusive_pix
        ] = img1
        self.img2 = self.img2.astype(np.uint8)

    def _cut_and_pool(self):
        '''
        画像を切り出しリストで保存
        '''
        self.img1_sub = []
        self.img2_sub = []
        self.img_index = []

        for j in range(self.len[1]):
            for i in range(self.len[0]):
                self.img1_sub.append(self.img1[
                    self.stride[0] * i:self.stride[0] * i + self.trimed_size[0],
                    self.stride[1] * j:self.stride[1] * j + self.trimed_size[1]
                ])
                self.img2_sub.append(self.img2[
                    self.stride[0] * i:self.stride[0] * i + self.trimed_size[0],
                    self.stride[1] * j:self.stride[1] * j + self.trimed_size[1]
                ])
                self.img_index.append([i, j])

    def _solver(self, solve_image, solve_template):
        """
        小画像に対しdeepmatchingを実施する
        """
        co_cls = Correlation_map(solve_image, solve_template, window_size=self.window_size, feature_name=self.feature_name)
        co_cls()
        if self.log_flg:
            print('complete to create multi-level correlation pyramid')
            print('pyramid level: {}, N={}'.format(co_cls.iteration, co_cls.N_map))
            self.log_flg = False

        cls = Matching(co_cls, sub_pix=self.sub_pix, filtering=self.filtering, filter_window_size=self.filtering_window_size, filtering_num=self.filtering_num)
        out = cls()

        del co_cls
        del cls

        # results = [Calc_difference.cal_map(out, mode=mode_here) for mode_here in self.degree_map_mode]

        # d_map = Calc_difference.cal_map(out, mode='elevation')
        """
        if self.sub_pix:
            direction_list = [1 if name == 'elevation' else 0 for name in self.degree_map_mode]
            return np.array([sub_pix_cal(Calc_difference.cal_map(out, mode=mode_here), out[2, :, :], direction=direction_list[i]) for (i, mode_here) in enumerate(self.degree_map_mode)]), out[2, :, :]
        else:
            return np.array([Calc_difference.cal_map(out, mode=mode_here) for mode_here in self.degree_map_mode]), out[2, :, :]
        """
        return np.array([Calc_difference.cal_map(out, mode=mode_here) for mode_here in self.degree_map_mode]), out[2, :, :]

    def _execute_matching(self):
        """
        小画像ごとにマッチングを行い結果を結合
        重なりなしのみ対応
        """
        len_list = [len(self.degree_map_mode)]
        size_list = [self.stride[i] * self.img_index[-1][i] + self.image_size[i] for i in range(2)]
        len_list.extend(size_list)
        """
        self.d_map = np.empty(
            len_list,
            dtype=np.uint8)
        """
        self.d_map = np.empty(
            len_list,
            dtype=float)
        self.out_map = np.empty(
            size_list,
            dtype=float)
        for idx in trange(len(self.img_index), desc='executing deepmatching'):
            i_here, j_here = self.img_index[idx]
            [
                self.d_map[
                    :,
                    self.stride[0] * i_here:self.stride[0] * i_here + self.image_size[0],
                    self.stride[1] * j_here:self.stride[1] * j_here + self.image_size[1]
                ],
                self.out_map[
                    self.stride[0] * i_here:self.stride[0] * i_here + self.image_size[0],
                    self.stride[1] * j_here:self.stride[1] * j_here + self.image_size[1]
                ]
            ] = self._solver(self.img1_sub[idx], self.img2_sub[idx])
            """
            print(self.stride[0] * i_here, self.stride[0] * i_here + self.image_size[0],
            self.stride[1] * j_here, self.stride[1] * j_here + self.image_size[1])
            """

    def __call__(self):
        self._cut_and_pool()
        self._execute_matching()
        return self.d_map, self.out_map

    @staticmethod
    def image_save(path, arr, threshold=[100, 190]):
        """
        numpy配列を画像として保存
        スレッショルド指定可能
        """
        # print(np.max(arr))
        # print(np.min(arr))
        arr = np.where(arr > threshold[1], threshold[1], arr)
        arr = np.where(arr < threshold[0], threshold[0], arr)
        pil_img = Image.fromarray(arr.astype(np.uint8))
        pil_img.save(path)


if __name__ == '__main__':
    """
    sanity check
    """
    import cv2

    """
    start = 100
    img1 = cv2.imread('./data/band3s.tif', cv2.IMREAD_GRAYSCALE)[start:start + 260, start:start + 260]
    img2 = cv2.imread('./data/band3bs.tif', cv2.IMREAD_GRAYSCALE)[start:start + 260, start:start + 260]

    cls = ImageCutSolver(img1, img2)
    cls._cut_and_pool()
    # cls._solver(cls.img1_sub[0], cls.img2_sub[0])
    cls._execute_matching()
    print(cls.d_map.shape)

    cv2.imwrite('./here.png', img1)
    cv2.imwrite('./output.png', cls.d_map * 30 + 100)
    """

    start_x = 1650
    start_y = 2500
    win_wid = 1000
    img_loaded = cv2.imread('./data/after-before-crossdis.tif')
    img1_raw = img_loaded[:, :, 1]  # 地震前
    img2_raw = img_loaded[:, :, 2]  # 地震後
    img3_raw = img_loaded[:, :, 0]  # 変化マップ

    img1 = img1_raw[start_y:start_y + win_wid, start_x:start_x + win_wid]
    img2 = img2_raw[start_y:start_y + win_wid, start_x:start_x + win_wid]
    img3 = img3_raw[start_y:start_y + win_wid, start_x:start_x + win_wid]

    cls = ImageCutSolver(img1, img2, degree_map_mode=['distance', 'elevation', 'elevation2'], window_size=15, image_size=[16, 16], stride=[12, 12])

    # cls.image_save('./here.png', img1)
    res_list, correlation_map = cls()
    for i, name in enumerate(['distance', 'elevation', 'elevation2']):
        cls.image_save('./output/igarss/' + name + '.png', res_list[i] * 90 + 100)
        np.save('./output/igarss/' + name, res_list[i])
    cls.image_save('./output/igarss/' + 'correlation' + '.png', correlation_map * 30 + 100)
    np.save('./output/igarss/' + 'correlation', correlation_map)
    exclusive_pix = [16, 16]
    ImageCutSolver.image_save('./output/igarss/here.png', img1[exclusive_pix[0]:-exclusive_pix[0], exclusive_pix[1]:-exclusive_pix[1]], threshold=[0, 255])
    ImageCutSolver.image_save('./output/igarss/here2.png', img2[exclusive_pix[0]:-exclusive_pix[0], exclusive_pix[1]:-exclusive_pix[1]], threshold=[0, 255])
    ImageCutSolver.image_save('./output/igarss/seikai.png', img3[exclusive_pix[0]:-exclusive_pix[0], exclusive_pix[1]:-exclusive_pix[1]], threshold=[0, 255])
