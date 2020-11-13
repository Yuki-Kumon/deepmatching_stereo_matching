# -*- coding: utf-8 -*-

"""
execute extended deepmatching with raw file input
Author :
    Yuki Kumon
Last Update :
    2020-01-18
"""


import numpy as np
import cv2

from absl import app, flags, logging
from absl.flags import FLAGS

from misc.image_cut_solver import ImageCutSolver
from misc.raw_read import RawRead


flags.DEFINE_string('image1_path', './data/TiffColor/Pakistan2013Before312.tif', '')
flags.DEFINE_string('image2_path', './data/TiffColor/Pakistan2013After312.tif', '')
flags.DEFINE_list('rates', '1, 1', '')

flags.DEFINE_list('cut_start', '2050, 1600', '')
flags.DEFINE_list('cut_size', '1000, 1000', '')

flags.DEFINE_integer('window_size', 15, '')
flags.DEFINE_list('crop_size', '64, 64', '')
flags.DEFINE_list('stride', '60, 60', '')
flags.DEFINE_bool('sub_pix', False, '')
flags.DEFINE_bool('filtering', False, '')
flags.DEFINE_integer('filtering_window_size', 3, '')
flags.DEFINE_integer('filtering_num', 4, '')
flags.DEFINE_string('filtering_mode', 'median', '')


def main(_argv):
    # set integer list
    rates = [int(x) for x in FLAGS.rates]
    cut_start = [int(x) for x in FLAGS.cut_start]
    cut_size = [int(x) for x in FLAGS.cut_size]

    crop_size = [int(x) for x in FLAGS.crop_size]
    stride = [int(x) for x in FLAGS.stride]

    DEGREE_NAMES = ['elevation', 'elevation2']

    # read image
    # image1 = RawRead.read(FLAGS.image1_path, rate=rates[0])
    # image2 = RawRead.read(FLAGS.image2_path, rate=rates[1])
    image1 = cv2.imread(FLAGS.image1_path)
    image2 = cv2.imread(FLAGS.image2_path)

    # crop images
    image1 = image1[cut_start[0]:cut_start[0] + cut_size[0], cut_start[1]:cut_start[1] + cut_size[1], :]
    image2 = image2[cut_start[0]:cut_start[0] + cut_size[0], cut_start[1]:cut_start[1] + cut_size[1], :]

    # print raw image
    ImageCutSolver.image_save('./output/rssj69/raw/here.png', image1, threshold=[0, 255])
    ImageCutSolver.image_save('./output/rssj69/raw/here2.png', image2, threshold=[0, 255])

    # execute deepmatching
    solver = ImageCutSolver(
        image1, image2,
        degree_map_mode=DEGREE_NAMES,
        window_size=FLAGS.window_size,
        image_size=crop_size,
        stride=stride,
        sub_pix=FLAGS.sub_pix,
        filtering=FLAGS.filtering,
        filtering_window_size=FLAGS.filtering_window_size,
        filtering_num=FLAGS.filtering_num,
        filtering_mode=FLAGS.filtering_mode,
        is_color=True,
    )

    res_list, correlation_map = solver()

    # save computation result as numpy array
    for i, name in enumerate(DEGREE_NAMES):
        np.save('./output/rssj69/raw/' + name, res_list[i])
    np.save('./output/rssj69/raw/' + 'correlation', correlation_map)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
