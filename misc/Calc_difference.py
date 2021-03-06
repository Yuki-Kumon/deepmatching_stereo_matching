# -*- coding: utf-8 -*-

"""
calculate difference map from matching map
Author :
    Yuki Kumon
Last Update :
    2019-09-26
"""


import sys

import numpy as np


class Calc_difference():
    '''
    Matchingクラスで計算したマップから視差マップを計算する
    '''

    def __init__(self):
        pass

    @staticmethod
    def cal_map(map, mode='elevation'):
        '''
        視差画像を計算する
        '''
        MODES = ['elevation', 'elevation2', 'distance']
        if mode not in MODES:
            print('please input valid mode! {} are ok. yours is \'{}\''.format(MODES, mode))
            sys.exit()
        # print('mode: {}'.format(mode))

        d_map = np.empty((map.shape[1], map.shape[2])).astype('float')
        if mode == 'elevation':
            for i in range(d_map.shape[0]):
                for j in range(d_map.shape[1]):
                    d_map[i, j] = float(j) - map[1, i, j]
        if mode == 'elevation2':
            for i in range(d_map.shape[0]):
                for j in range(d_map.shape[1]):
                    d_map[i, j] = float(i) - map[0, i, j]
        elif mode == 'distance':
            for i in range(d_map.shape[0]):
                for j in range(d_map.shape[1]):
                    d_map[i, j] = float(np.linalg.norm(np.array([i, j]).astype('float') - map[:2, i, j]))
        return d_map


if __name__ == '__main__':
    """
    sanity check
    """

    """
    from Correlation_map import Correlation_map
    from Matching import Matching

    import cv2

    # atomicな特徴マップが一辺が2^nじゃないとバグるカス実装です。。。
    start = 100
    img1 = cv2.imread('./data/band3s.tif', cv2.IMREAD_GRAYSCALE)[start:start + 68, start:start + 260]
    img2 = cv2.imread('./data/band3bs.tif', cv2.IMREAD_GRAYSCALE)[start:start + 68, start:start + 260]

    # co_cls = Correlation_map(img1, img2, window_size=5)
    co_cls = Correlation_map(img1, img2, window_size=5, feature_name='cv2.TM_CCOEFF')
    co_cls()

    cls = Matching(co_cls)
    out = cls()

    # d_map = Calc_difference.cal_map(out, mode='elevation')
    d_map = Calc_difference.cal_map(out, mode='distance')

    cv2.imwrite('./output.png', d_map * 30 + 100)
    cv2.imwrite('here.png', img1)
    """

    from Correlation_map_v2 import CorrelationMapV2
    from Matching_v2 import MatchingV2

    import cv2

    # atomicな特徴マップが一辺が2^nじゃないとバグるカス実装です。。。
    start = 100
    img1 = cv2.imread('./data/band3s.tif', cv2.IMREAD_GRAYSCALE)[start:start + 258, start:start + 258]
    img2 = cv2.imread('./data/band3bs.tif', cv2.IMREAD_GRAYSCALE)[start:start + 258, start:start + 258]

    # co_cls = Correlation_map(img1, img2, window_size=5)
    co_cls = CorrelationMapV2(img1, img2)
    co_cls()

    cls = MatchingV2(co_cls)
    out = cls()

    print(cls.obj.co_map_list[-1])
    print(cls.obj.co_map_list[-2])

    d_map = Calc_difference.cal_map(out, mode='elevation')
    # d_map = Calc_difference.cal_map(out, mode='distance')

    print(d_map)
    d_map2 = np.empty((out.shape[1], out.shape[2])).astype('int64')
    for i in range(d_map2.shape[0]):
        for j in range(d_map2.shape[1]):
            d_map2[i, j] = i - out[0, i, j]
    print(d_map2)
    cv2.imwrite('./output.png', (d_map + 8) * 30 + 100)
    cv2.imwrite('here.png', img1)
