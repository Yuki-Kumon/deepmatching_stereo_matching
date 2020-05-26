# -*- coding: utf-8 -*-

"""
numpy arrayを画像へと変換する。
Author :
    Yuki Kumon
Last Update :
    2020-05-18
"""


import numpy as np
import os

from misc.visualizer import Visualizer
from misc.image_cut_solver import ImageCutSolver

from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_list('array_path', './output/igarss/raw/elevation.npy, ./output/igarss/raw/elevation2.npy', 'numpy array path list')
flags.DEFINE_list('strech_range', '-1.5, 5.5, -5.5, 1.5', 'strech range for input array')  # sub_pixでは -5.5,1.5,-8,3 にしてコメントアウトを外す
flags.DEFINE_string('output_path', './output/igarss/raw/temp_out', 'output_path')
flags.DEFINE_list('output_name', 'elevation.png, elevation2.png', 'output file name')


def main(_argv):
    '''
    main function
    '''
    # make dir
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    for i, path in enumerate(FLAGS.array_path):
        # load numpy array
        d_map = np.load(path)
        d_map = d_map.astype(float)
        print('===file {} is loaded==='.format(path))
        print('===original data===')
        print('min', np.min(d_map))
        print('max', np.max(d_map))
        print('mean', np.mean(d_map))

        # conver
        range = [float(x) for x in FLAGS.strech_range[2 * i: 2 * i + 2]]
        d_map = Visualizer.strech_for_write(d_map, range=range)
        # d_map = 255 - Visualizer.strech_for_write(d_map, range=range)
        print('===streched data===')
        print('min', np.min(d_map))
        print('max', np.max(d_map))
        print('mean', np.mean(d_map))

        # save
        ImageCutSolver.image_save(os.path.join(FLAGS.output_path, FLAGS.output_name[i]), d_map, threshold=[0, 255])


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
