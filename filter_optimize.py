# -*- coding: utf-8 -*-

"""
optimize loop for sub pix
Author :
    Yuki Kumon
Last Update :
    2020-01-13
"""


from model.optimize_loop import optimize_loop
from misc.image_cut_solver import ImageCutSolver
from misc.Calc_difference import Calc_difference

import numpy as np

from absl import app, flags, logging
from absl.flags import FLAGS
from tqdm import trange

flags.DEFINE_string('elevation_path', './output/igarss/elevation.npy', '')
flags.DEFINE_string('elevation2_path', './output/igarss/elevation2.npy', '')
flags.DEFINE_string('co_path', './output/igarss/correlation.npy', '')
flags.DEFINE_float('allowed_error', 1000 * 1000 * 0.005, '')
flags.DEFINE_float('alpha', 0.008, '')


def main(_argv):
    # load array
    d_map1 = np.load(FLAGS.elevation_path)
    d_map2 = np.load(FLAGS.elevation2_path)

    d_map1 = d_map1.astype(float)
    d_map2 = d_map2.astype(float)

    co_map = np.load(FLAGS.co_path)[2]

    loop_limit = 20
    size = d_map1.shape
    # execute
    for loop in trange(loop_limit, desc='filter_optimize'):
        d_map1, error = optimize_loop(d_map1, co_map, FLAGS.alpha, 1, size)
        print(error)
        if error < FLAGS.allowed_error:
            break
    for loop in trange(loop_limit, desc='filter_optimize'):
        d_map2, error = optimize_loop(d_map1, co_map, FLAGS.alpha, 1, size)
        print(error)
        if error < FLAGS.allowed_error:
            break
    # write output

    ImageCutSolver.image_save('./output/igarss/' + 'elevation_optimized' + '.png', d_map1 * 30 + 100)
    ImageCutSolver.image_save('./output/igarss/' + 'elevation2_optimized' + '.png', d_map2 * 30 + 100)
    # ImageCutSolver.image_save('./output/igarss/' + 'distance_optimized' + '.png', d_map3 * 30 + 100)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
