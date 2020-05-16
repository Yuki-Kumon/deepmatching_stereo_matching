# -*- coding: utf-8 -*-

"""
可視化ツール
Author :
    Yuki Kumon
Last Update :
    2020-05-07
"""


import numpy as np


class Visualizer():
    '''
    画像の書き出し用のクラス
    '''

    def __init__(self):
        pass

    @staticmethod
    def threshold(image, range=[-2, 2]):
        '''
        指定範囲を超えた画素値を足切りする
        image: numpy array
        '''
        range_f = [float(x) for x in range]
        image = np.where(image < range_f[0], range_f[0], image)
        image = np.where(image > range_f[1], range_f[1], image)

        return image

    @staticmethod
    def translation(image, range=[-2, 2]):
        '''
        書き出しように変換
        '''
        range_f = [float(x) for x in range]
        return ((image + range_f[0]) * (255 / np.sum(range_f))).astype('uint8')

    @classmethod
    def strech_for_write(self, image, range=[-2, 2]):
        '''
        画像として正しく書き出せるようにストレッチする
        '''
        image = self.threshold(image, range)
        image = self.translation(image, range)

        return image


if __name__ == '__main__':
    """
    sanity check
    """
    import sys
    sys.path.append('.')
    from misc.image_cut_solver import ImageCutSolver

    # load numpy array
    d_map1 = np.load('./output/igarss/raw/setup1_band2/elevation.npy')
    d_map1 = d_map1.astype(float)
    print('min', np.min(d_map1))
    print('max', np.max(d_map1))
    print('mean', np.mean(d_map1))

    # 処理
    d_map1 = Visualizer.strech_for_write(d_map1, range=[-2, 5])
    print('min', np.min(d_map1))
    print('max', np.max(d_map1))
    print('mean', np.mean(d_map1))

    ImageCutSolver.image_save('./test.png', d_map1, threshold=[0, 255])
