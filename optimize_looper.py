# -*- coding: utf-8 -*-

"""
卒論の手法を利用し、ループ計算により誤差を減らす
Author :
    Yuki Kumon
Last Update :
    2020-05-04
"""

import sys
sys.path.append('.')

import numpy as np

from absl import app, flags, logging
from absl.flags import FLAGS
from tqdm import trange

from misc.image_cut_solver import ImageCutSolver
from misc.opt_loop import *


flags.DEFINE_string('elevation_path', './output/igarss/elevation.npy', '')
flags.DEFINE_string('elevation2_path', './output/igarss/elevation2.npy', '')
flags.DEFINE_string('co_path', './output/igarss/correlation.npy', '')
flags.DEFINE_integer('exclusion', 2, '')
flags.DEFINE_list('sigma', [1.5, 1.0], '')
flags.DEFINE_float('allowed_error', 1000 * 1000 * 0.005, '')
flags.DEFINE_float('alpha', 0.008, '')


def main(_argv):
    # load array
    d_map1 = np.load(FLAGS.elevation_path)
    d_map2 = np.load(FLAGS.elevation2_path)

    d_map1 = d_map1.astype(float)
    d_map2 = d_map2.astype(float)

    co_map = np.load(FLAGS.co_path)

    loop_limit = 20
    size = d_map1.shape
    # print(d_map1.shape)
    # print(co_map.shape)
    sigma = np.array([int(x) for x in FLAGS.sigma])

    # 重み付け配列の作成
    gausian_weight, color_weight = make_weight(d_map1, FLAGS.exclusion, size, sigma)

    # elevation(横方向)について、ループ計算を実施する
    for loop in trange(loop_limit, desc='filter_optimize'):
        d_map1, error = optimize_loop_bilateral_horizon(d_map1, color_weight, gausian_weight, co_map, FLAGS.alpha, FLAGS.exclusion, size)
        print(error)
        if error < FLAGS.allowed_error:
            break



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
