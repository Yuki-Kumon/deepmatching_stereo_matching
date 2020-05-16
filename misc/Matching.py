# -*- coding: utf-8 -*-

"""
matching on correlation map
Author :
    Yuki Kumon
Last Update :
    2019-09-24
"""


import sys

import cv2
import numpy as np
import torch
import torch.nn as nn


class Matching():
    '''
    multi-level correlation pyramidからマッチングを行う
    原著の14式に従って計算していく
    '''

    def __init__(
        self,
        Co_obj=None,
        filter_window_size=5,
        filtering=False,
        filtering_num=3,
        sub_pix=True
    ):
        try:
            Co_obj.co_map_list
        except AttributeError as e:
            print('Error!: {}'.format(e))
            print('please run \'obj=Correlation_map()\' and \'obj()\' first.')
            sys.exit()

        self.obj = Co_obj
        self.Padding = Zero_padding()
        self.Padding.eval()

        self.filtering_num = filtering_num
        self.filter_window_size = filter_window_size
        self.filtering = filtering
        self.sub_pix = sub_pix

    def _calc_near_match(self, co_map, p, p_dot):
        '''
        pとp_dotはco_mapの解像度に合わせた座標
        p_iとして共に少し4つのchildrenとして移動させた上でこの関数に読ませることにする
        原著の13式に従って対応点を計算する
        13式のmを計算する
        '''
        # pに対応した特徴マップを取り出す
        map_on_p = co_map[p[0], p[1]]
        # 端対策にゼロパディングする
        map_on_p_padded = self.Padding(torch.from_numpy(map_on_p[None])).numpy()[0]
        # p_dot周辺の3×3を取り出す
        co_map_near_p_dot = map_on_p_padded[p_dot[0] - 1 + 1:p_dot[0] + 2 + 1, p_dot[1] - 1 + 1:p_dot[1] + 2 + 1]
        # この中で相関値最大の座標を計算
        m = np.unravel_index(np.argmax(co_map_near_p_dot), co_map_near_p_dot.shape)
        # 全部0だった場合の処理(もし最大値がだいたいゼロなら、移動していないとみなす)
        if np.max(co_map_near_p_dot) < 0.0001:
            m = [1, 1]
        # p_dot, 新たな相関値を返す
        # print(p_dot[0] - 1 + 1, p_dot[0] + 2 + 1, p_dot[1] - 1 + 1, p_dot[1] + 2 + 1, co_map_near_p_dot, co_map_near_p_dot)
        return p_dot[0] + m[0] - 1, p_dot[1] + m[1] - 1, co_map_near_p_dot[m[0], m[1]] + map_on_p[p_dot[0], p_dot[1]]

    def _initial_move_map(self):
        '''
        ピラミッドの頂点での動きのマップを計算する
        '''
        # 各ピクセルに(対応するy, 対応するx, 相関値s)が格納されている
        map = np.zeros((3, self.obj.co_map_list[-1].shape[0], self.obj.co_map_list[-1].shape[1]))
        for i in range(map.shape[1]):
            for j in range(map.shape[2]):
                map[:2, i, j] = i, j
                # 初めの移動量を計算(これは必要なさそうではある)
                map[:, i, j] = self._calc_near_match(self.obj.co_map_list[-1], (i, j), map[:2, i, j].astype('int64'))
        if self.filtering and self.filtering_num > 0:
            map = self._filter(map)
            self.filtering_num -= 1
        self.map = map
        self.map_idx = -1
        self.N = self.obj.N_map

    def _B(self):
        '''
        原著の14式
        各パッチの左上の座標を用いて計算する
        '''
        # Nとiterarionとmapはこの後更新する。N > 0でwhileループで回せばいいと思う
        # N = self.obj.N_map
        N = self.N
        map_idx = self.map_idx - 1
        map_here = self.map
        map_updated = np.empty((3, map_here.shape[1] * 2, map_here.shape[2] * 2))

        # 式中のベクトルo
        o_list = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])

        for i in range(map_here.shape[1]):
            for j in range(map_here.shape[2]):
                # 対応するquadrantの一つ上での解像度での座標を計算しておく
                p_upper_left = np.array([i * 2, j * 2])
                p_dot_upper_left = (map_here[:2, i, j] * 2).astype('int64')
                # ここでの相関値を取り出しておく
                # s_here = self.map[2, i, j]
                # qauddrantごとに14式の計算を行い、mapを更新する
                for o_idx in range(4):
                    o_here = o_list[o_idx]
                    p_here = p_upper_left + o_here
                    p_dot_here = p_dot_upper_left + o_here
                    # 13式に従い、mを計算、mapを更新する
                    map_updated[:, p_here[0], p_here[1]] = self._calc_near_match(
                        self.obj.co_map_list[map_idx],
                        (p_here[0], p_here[1]),
                        (p_dot_here[0], p_dot_here[1])
                    )
                    # print(map_updated[:, p_here[0], p_here[1]])
        # 諸々の値を更新
        self.map_idx -= 1
        self.N = int(N / 2)
        del self.map
        if self.filtering and self.filtering_num > 0:
            map_updated = self._filter(map_updated)
            self.filtering_num -= 1
        self.map = map_updated

    def _calc_match(self):
        '''
        原著のB式を繰り返し用いてマッチングを計算する
        '''
        # N > 0であれば計算を行う
        while 1:
            self._B()
            if self.N == 1:
                break

    def _sub_pix_cal(self):
        '''
        サブピクセル近似
        最終的なマッチング結果と最初に計算した相関マップを用いる
        '''
        co_map_last = self.obj.co_map_list[0]
        for i in range(self.map.shape[1]):
            for j in range(self.map.shape[2]):
                corresponding = [int(i) for i in self.map[:2, i, j]]
                # 縦について近傍の点から小数のずれを計算
                d_x = i - self.map[0, i, j]
                try:
                    r0 = co_map_last[i, j, corresponding[0], corresponding[1]]
                    r1 = co_map_last[i, j, corresponding[0] + 1, corresponding[1]]
                    r_ = co_map_last[i, j, corresponding[0] - 1, corresponding[1]]
                    self.map[0, i, j] = i + d_x - (r1 - r_) / (2 * (r1 + r_ - 2 * r0))
                except:
                    self.map[0, i, j] = i + d_x
                # 横について近傍の点から小数のずれを計算
                d_y = j - self.map[1, i, j]
                try:
                    r0 = co_map_last[i, j, corresponding[0], corresponding[1]]
                    r1 = co_map_last[i, j, corresponding[0], corresponding[1] + 1]
                    r_ = co_map_last[i, j, corresponding[0], corresponding[1] - 1]
                    self.map[1, i, j] = j + d_y - (r1 - r_) / (2 * (r1 + r_ - 2 * r0))
                except:
                    self.map[1, i, j] = j + d_y

    def __call__(self):
        '''
        multi-level correlation pyramidからマッチング計算を行う
        '''
        self._initial_move_map()
        # print('complete to create initial matching map')
        self._calc_match()
        # print('complete backtracking')
        if self.sub_pix:
            self._sub_pix_cal()

        return self.map

    def _filter(self, map_here):
        """
        飛び値を無視する
        """
        # mapのサイズを調べておく(paddingなしで処理を行うので)
        map_shape = map_here.shape
        if map_shape[1] >= self.filter_window_size and map_shape[2] >= self.filter_window_size:
            # 除外ピクセル数
            exclusive_pix = int((self.filter_window_size - 1) / 2)
            # 処理
            # 移動量マップ用の配列
            d_map = np.empty((map_shape[1], map_shape[1])).astype('int64')
            d_map2 = np.empty((map_shape[1], map_shape[1])).astype('int64')
            for i in range(d_map.shape[0]):
                for j in range(d_map.shape[1]):
                    # 移動量マップへ変換
                    d_map[i, j] = map_here[1, i, j] - j
                    d_map2[i, j] = map_here[0, i, j] - i
            for i in range(exclusive_pix, map_shape[1] - exclusive_pix):
                for j in range(exclusive_pix, map_shape[2] - exclusive_pix):
                    # 近傍のピクセルの平均値で更新する
                    map_here[1, i, j] = round(np.mean(d_map[i - exclusive_pix:i + exclusive_pix + 1, j - exclusive_pix:j + exclusive_pix + 1])) + j
                    map_here[0, i, j] = round(np.mean(d_map2[i - exclusive_pix:i + exclusive_pix + 1, j - exclusive_pix:j + exclusive_pix + 1])) + i
        return map_here


class Zero_padding(nn.Module):
    '''
    原著の14式の計算のため、ゼロパディングしておく
    '''

    def __init__(self):
        super(Zero_padding, self).__init__()
        self.m = nn.ZeroPad2d(1)

    def forward(self, x):
        return self.m(x)


if __name__ == '__main__':
    """
    sanity check
    """
    from Correlation_map import Correlation_map

    # atomicな特徴マップが一辺が2^nじゃないとバグるカス実装です。。。
    img1 = cv2.imread('./data/band3s.tif', cv2.IMREAD_GRAYSCALE)[500:500 + 68, 500:500 + 260]
    img2 = cv2.imread('./data/band3bs.tif', cv2.IMREAD_GRAYSCALE)[500:500 + 68, 500:500 + 260]

    co_cls = Correlation_map(img1, img2, window_size=5)
    co_cls()

    # 試しに書き出し
    """
    cv2.imwrite('out.png', co_cls.co_map_list[1][0, 0] * 50)
    cv2.imwrite('out0.png', co_cls.co_map_list[0][0, 0] * 50)
    cv2.imwrite('here.png', img1)
    """

    cls = Matching(co_cls)
    # cls._initial_move_map()
    # cls._B()
    # cls._calc_match()
    out = cls()
