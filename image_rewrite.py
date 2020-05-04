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
from misc.optimize_loop import image_threshold


def rewrite(path, nega=False, strech=[50, 100]):
    np_path = path
    # load numpy array
    arr = np.load(np_path)

    # strech
    arr = image_threshold(arr, threshold=[-2, 3])
    print(np.mean(arr))
    rate = strech[0]
    ratio = -rate if nega else rate
    arr = arr * ratio + strech[1]

    # write
    # print(os.path.splitext(np_path))
    ImageCutSolver.image_save(os.path.splitext(np_path)[0] + '.png', arr, threshold=[10, 250])


def _rename(path, setup):
    ELEVS = ['elevation.png', 'elevation2.png']
    for here in ELEVS:
        os.rename(os.path.join(path, here), os.path.join(path, setup + '_' + here))


def _generater(path='./output/igarss/raw/'):
    '''
    パスを生成
    path: './output/igarss/raw/'
    '''
    ELEVS = ['elevation.npy', 'elevation2.npy']
    NEGAS = [True, False]
    NEGAS = [False, True]

    return [os.path.join(path, here) for here in ELEVS], NEGAS


def executer(path, setup, strech=[80, 80]):
    path_list, nega_list = _generater(path)
    for path_here, nega in zip(path_list, nega_list):
        rewrite(path_here, nega=nega, strech=strech)
    _rename(path, setup)


if __name__ == '__main__':

    root = './output/igarss/raw'
    executer(root, 'setup1')
    """
    s_1 = [80, 80]
    s_2 = [80, 80]
    setup = 'setup1'
    rewrite('./output/igarss/' + setup + '/elevation.npy', nega=False, strech=s_1)
    rewrite('./output/igarss/' + setup + '/elevation2.npy', nega=True, strech=s_2)
    _rename('./output/igarss/' + setup, setup)

    setup = 'setup1_sub'
    rewrite('./output/igarss/' + setup + '/elevation.npy', nega=True, strech=s_1)
    rewrite('./output/igarss/' + setup + '/elevation2.npy', nega=False, strech=s_2)
    _rename('./output/igarss/' + setup, setup)

    setup = 'setup2_sub'
    rewrite('./output/igarss/' + setup + '/elevation.npy', nega=True, strech=s_1)
    rewrite('./output/igarss/' + setup + '/elevation2.npy', nega=False, strech=s_2)
    _rename('./output/igarss/' + setup, setup)

    setup = 'setup3_sub'
    rewrite('./output/igarss/' + setup + '/elevation.npy', nega=True, strech=s_1)
    rewrite('./output/igarss/' + setup + '/elevation2.npy', nega=False, strech=s_2)
    _rename('./output/igarss/' + setup, setup)
    """
