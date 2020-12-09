# -*- coding: utf-8 -*-

"""
matching on correlation map
Author :
    Yuki Kumon
Last Update :
    2020-05-18
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
        filter_window_size=3,
        filtering=False,
        filtering_num=3,
        filtering_mode='median',
        sub_pix=False,
        optim_mode=None,  # 1だとsub_pix、2だとoptim_loop(単純平均)
        loop_limit=20,
        allowed_error=32 * 32 * 0.005,
        gaus_alpha=0.008,
        # sub_pix_range=[-0.5, 0.5]
    ):
        try:
            Co_obj.co_map_list
        except AttributeError as e:
            print('Error!: {}'.format(e))
            print('please run \'obj=Correlation_map()\' and \'obj()\' first.')
            sys.exit()

        MODES = ['average', 'median']
        assert filtering_mode in MODES, 'invalid filtering mode is input!: {}'.format(filtering_mode)

        self.obj = Co_obj
        self.Padding = Zero_padding()
        self.Padding.eval()

        self.filtering_num = filtering_num
        self.filter_window_size = filter_window_size
        self.filtering = filtering
        self.filtering_mode = filtering_mode

        self.optim_mode = 0
        if sub_pix:
            self.optim_mode = 1  # 単なるサブピクセル近似、後方互換性のため引数を残している
        elif not sub_pix and optim_mode:
            self.optim_mode = int(optim_mode)

        self.loop_limit = loop_limit
        self.allowed_error = allowed_error
        self.gaus_alpha = gaus_alpha

        # self._sub_pix_range = sub_pix_range

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

    """
    @classmethod
    def _sub_pix_thred(self, val):
        '''
        小数精度近似の足切り
        '''
        if val > self.sub_pix_range[0]:
            val = self.sub_pix_range[0]
        if val < self.sub_pix_range[1]:
            val = self.sub_pix_range[1]

        return val
    """

    @classmethod
    def _sub_pix_compute(self, r0, r1, r_):
        '''
        二次関数近似の計算
        二次関数近似を適用できない箇所には適用しない
        '''
        if r0 > r1 and r0 > r_:
            diff = - (r1 - r_) / (2 * (r1 + r_ - 2 * r0))
        else:
            diff = 0
        return diff

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

                    # diff = - (r1 - r_) / (2 * (r1 + r_ - 2 * r0))
                    diff = self._sub_pix_compute(r0, r1, r_)
                    self.map[0, i, j] = i - d_x + diff
                except:
                    self.map[0, i, j] = i - d_x
                # 横について近傍の点から小数のずれを計算
                d_y = j - self.map[1, i, j]
                try:
                    r0 = co_map_last[i, j, corresponding[0], corresponding[1]]
                    r1 = co_map_last[i, j, corresponding[0], corresponding[1] + 1]
                    r_ = co_map_last[i, j, corresponding[0], corresponding[1] - 1]

                    # diff = - (r1 - r_) / (2 * (r1 + r_ - 2 * r0))
                    diff = self._sub_pix_compute(r0, r1, r_)
                    self.map[1, i, j] = j - d_y + diff
                except:
                    self.map[1, i, j] = j - d_y

    def _sub_pix_optim(self):
        '''
        サブピクセル近似
        卒論の手法を使う
        小領域内のみでエネルギーを計算する
        '''
        # img_disをサブピクセルの値に変換
        self._sub_pix_cal()
        # 縦方向
        img_dis = self._map_to_diff_vertical(self.map[0])

        for loop in range(self.loop_limit):
            img_dis, error = self._optimize_loop_vertical(img_dis)
            # print(error)
            if(error < self.allowed_error):
                break

        self.map[0] = self._diff_to_map_vertical(img_dis)

        # 横方向
        img_dis = self._map_to_diff_horizontal(self.map[1])
        for loop in range(self.loop_limit):
            img_dis, error = self._optimize_loop_vertical(img_dis)
            if(error < self.allowed_error):
                break

        self.map[1] = self._diff_to_map_horizontal(img_dis)

    @staticmethod
    def _map_to_diff_vertical(res_map):
        diff = np.empty_like(res_map)
        for i in range(res_map.shape[0]):
            for j in range(res_map.shape[1]):
                diff[i, j] = i - res_map[i, j]
        return diff

    @classmethod
    def _map_to_diff_horizontal(self, res_map):
        return self._map_to_diff_vertical(res_map.T)

    @staticmethod
    def _diff_to_map_vertical(diff_map):
        res = np.empty_like(diff_map)
        for i in range(diff_map.shape[0]):
            for j in range(diff_map.shape[1]):
                res[i, j] = i - diff_map[i, j]
        return res

    @classmethod
    def _diff_to_map_horizontal(self, diff_map):
        return self._diff_to_map_vertical(diff_map).T

    def _optimize_loop_vertical(self, img_dis):
        '''
        卒論の手法で最適化を実施する
        単純平均フィルター版
        縦方向(第一引数)での最適化
        窓のサイズは3×3で固定(sum_dの書き換えがめんどくさいため)
        TODO: 卒論のコードはb_iの部分を間違えてるな！img_dis[i, j]→b(二次関数近似の答え)[i, j]にする！！！
        '''
        coefficient = self.obj.co_map_list[0]
        alpha = self.gaus_alpha

        error = 0
        for i in range(1, img_dis.shape[0] - 1 - 1):
            for j in range(1, img_dis.shape[1] - 1 - 1):
                corresponding = [int(i) for i in self.map[:2, i, j]]
                try:
                    sum_d = img_dis[i, j - 1] + img_dis[i, j + 1] + img_dis[i - 1, j] + img_dis[i + 1, j]
                    a = coefficient[i, j, corresponding[0], corresponding[1]]
                    d_new = (-a * img_dis[i, j] + alpha * sum_d) / (-a + 4.0 * alpha)
                    img_dis[i, j] = d_new
                except:
                    img_dis[i, j] = img_dis[i, j]
        # 逆から
        for i in range(1, img_dis.shape[0] - 1 - 1):
            for j in range(1, img_dis.shape[1] - 1 - 1):
                i = img_dis.shape[0] - i - 1
                j = img_dis.shape[1] - j - 1
                corresponding = [int(i) for i in self.map[:2, i, j]]
                try:
                    sum_d = img_dis[i, j - 1] + img_dis[i, j + 1] + img_dis[i - 1, j] + img_dis[i + 1, j]
                    a = coefficient[i, j, corresponding[0], corresponding[1]]
                    d_new = (-a * img_dis[i, j] + alpha * sum_d) / (-a + 4.0 * alpha)
                    error += abs(img_dis[i, j] - d_new)
                    img_dis[i, j] = d_new
                except:
                    img_dis[i, j] = img_dis[i, j]
        return img_dis, error

    def _optimize_loop_horizontal(self, img_dis):
        return self._optimize_loop_vertical(img_dis)

    def __call__(self):
        '''
        multi-level correlation pyramidからマッチング計算を行う
        '''
        self._initial_move_map()
        # print('complete to create initial matching map')
        self._calc_match()
        # print('complete backtracking')
        '''
        if self.sub_pix:
            self._sub_pix_cal()
        '''
        if self.optim_mode == 1:
            # サブピクセル近似
            self._sub_pix_cal()
        elif self.optim_mode == 2:
            # 最適化(単純平均フィルター)
            self._sub_pix_optim()

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

            if self.filtering_mode == 'average':
                for i in range(exclusive_pix, map_shape[1] - exclusive_pix):
                    for j in range(exclusive_pix, map_shape[2] - exclusive_pix):
                        # 近傍のピクセルの平均値で更新する
                        map_here[1, i, j] = round(np.mean(d_map[i - exclusive_pix:i + exclusive_pix + 1, j - exclusive_pix:j + exclusive_pix + 1])) + j
                        map_here[0, i, j] = round(np.mean(d_map2[i - exclusive_pix:i + exclusive_pix + 1, j - exclusive_pix:j + exclusive_pix + 1])) + i
            elif self.filtering_mode == 'median':
                for i in range(exclusive_pix, map_shape[1] - exclusive_pix):
                    for j in range(exclusive_pix, map_shape[2] - exclusive_pix):
                        # 近傍のピクセルの中央値で更新する
                        map_here[1, i, j] = round(np.median(d_map[i - exclusive_pix:i + exclusive_pix + 1, j - exclusive_pix:j + exclusive_pix + 1])) + j
                        map_here[0, i, j] = round(np.median(d_map2[i - exclusive_pix:i + exclusive_pix + 1, j - exclusive_pix:j + exclusive_pix + 1])) + i
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
