# -*- coding: utf-8 -*-

"""
matching on correlation map
Author :
    Yuki Kumon
Last Update :
    2019-12-12
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

    def __init__(self, Co_obj=None):
        try:
            Co_obj.co_map_list
        except AttributeError as e:
            print('Error!: {}'.format(e))
            print('please run \'obj=Correlation_map()\' and \'obj()\' first.')
            sys.exit()

        self.obj = Co_obj
        self.Padding = Zero_padding()
        self.Padding.eval()

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
        # p_dot周辺の3×3を取り出す(座標のずれに注意)
        co_map_near_p_dot = map_on_p_padded[p_dot[0] - p[0] - 1 + 1:p_dot[0] - p[0] + 2 + 1, p_dot[1] - p[1] - 1 + 1:p_dot[1] - p[1] + 2 + 1]
        # この中で相関値最大の座標を計算
        m = np.unravel_index(np.argmax(co_map_near_p_dot), co_map_near_p_dot.shape)
        # 全部0だった場合の処理(もし最大値がだいたいゼロなら、移動していないとみなす)
        if np.max(co_map_near_p_dot) < 0.0001:
            m = [1, 1]
        # p_dot, 新たな相関値を返す
        # 相関値を足す際、p_dotの座標のズレに注意
        return p_dot[0] + m[0] - 1, p_dot[1] + m[1] - 1, co_map_near_p_dot[m[0], m[1]] + map_on_p[p_dot[0] - p[0], p_dot[1] - p[1]]

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
                # print(map[:, i, j])
                map[:, i, j] = self._calc_near_match(self.obj.co_map_list[-1], (i, j), map[:2, i, j].astype('int64'))
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

    def __call__(self):
        '''
        multi-level correlation pyramidからマッチング計算を行う
        '''
        self._initial_move_map()
        print('complete to create initial matching map')
        self._calc_match()
        print('complete backtracking')

        return self.map


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
    from Correlation_map_v2 import CorrelationMapV2

    # atomicな特徴マップが一辺が2^nじゃないとバグるカス実装です。。。
    img1 = cv2.imread('./data/band3s.tif', cv2.IMREAD_GRAYSCALE)[500:500 + 258, 500:500 + 258]
    img2 = cv2.imread('./data/band3bs.tif', cv2.IMREAD_GRAYSCALE)[500:500 + 258, 500:500 + 258]
    print(img1.shape)

    co_cls = CorrelationMapV2(img1, img2)
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
    print(out)
