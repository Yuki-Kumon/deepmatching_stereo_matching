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
import cv2

from absl import app, flags, logging
from absl.flags import FLAGS
from tqdm import trange

# from misc.image_cut_solver import ImageCutSolver
from image_rewrite import rewrite
from misc.opt_loop import *


flags.DEFINE_string('elevation_path', './output/igarss/raw/setup1_band2/elevation.npy', '')
flags.DEFINE_string('elevation2_path', './output/igarss/raw/setup1_band2/elevation2.npy', '')
flags.DEFINE_string('co_path', './output/igarss/raw/setup1_band2/correlation.npy', '')
flags.DEFINE_string('elevation_save_path', './output/igarss/raw/setup1_band2/elevation_opt.npy', '')
flags.DEFINE_string('elevation2_save_path', './output/igarss/raw/setup1_band2/elevation2_opt.npy', '')
flags.DEFINE_integer('exclusion', 3, '')
flags.DEFINE_list('sigma', [5.0, 5.0], '')
flags.DEFINE_float('allowed_error', 1000 * 1000 * 0.03, '')
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
    print('d_map1: ', np.mean(d_map1))
    print('d_map2: ', np.mean(d_map2))

    d_map2 += 2
    sigma = np.array([int(x) for x in FLAGS.sigma])

    if 0:
        # 重み付け配列の作成
        gausian_weight, color_weight = make_weight(d_map1, FLAGS.exclusion, size, sigma)

        # elevation(横方向)について、ループ計算を実施する
        for loop in trange(loop_limit, desc='filter_optimize'):
            d_map1, error = optimize_loop_bilateral_horizon(d_map1, color_weight, gausian_weight, co_map, FLAGS.alpha, FLAGS.exclusion, size)
            print(error)
            if error < FLAGS.allowed_error:
                break

        # 重み付け配列の作成
        gausian_weight, color_weight = make_weight(d_map2, FLAGS.exclusion, size, sigma)

        # elevation2(縦方向)について、ループ計算を実施する
        for loop in trange(loop_limit, desc='filter_optimize'):
            d_map1, error = optimize_loop_bilateral_vertical(d_map2, color_weight, gausian_weight, co_map, FLAGS.alpha, FLAGS.exclusion, size)
            print(error)
            if error < FLAGS.allowed_error:
                break
    else:
        d_map1 = cv2.bilateralFilter(d_map1.astype('uint8'), FLAGS.exclusion * 2 + 1, sigma[0], sigma[1])
        d_map2 = cv2.bilateralFilter(d_map2.astype('uint8'), FLAGS.exclusion * 2 + 1, sigma[0], sigma[1])

    print('d_map1: ', np.mean(d_map1))
    print('d_map2: ', np.mean(d_map2))
    # save as a numpy array
    np.save(FLAGS.elevation_save_path, d_map1)
    # save as an image
    rewrite(FLAGS.elevation_save_path, nega=False, strech=[80, 80])
    # save as a numpy array
    np.save(FLAGS.elevation2_save_path, d_map2 - 0.5)
    # save as an image
    rewrite(FLAGS.elevation2_save_path, nega=True, strech=[80, 80])


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
