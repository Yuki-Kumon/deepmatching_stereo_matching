# -*- coding: utf-8 -*-

"""
numpy arrayを読み込んで画像を書き出し直す
Author :
    Yuki Kumon
Last Update :
    2020-01-13
"""


import os

import numpy as np

from misc.image_cut_solver import ImageCutSolver


def rewrite(path):
    np_path = path
    # load numpy array
    arr = np.load(np_path)

    # strech
    arr = arr * 90 + 100

    # write
    # print(os.path.splitext(np_path))
    ImageCutSolver.image_save(os.path.splitext(np_path)[0] + '.png', arr, threshold=[10, 250])


if __name__ == '__main__':
    rewrite('./output/igarss/setup3/elevation.npy')
    rewrite('./output/igarss/setup3/elevation2.npy')
