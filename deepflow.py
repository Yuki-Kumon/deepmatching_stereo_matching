import argparse
import os
import cv2
import numpy as np

from misc.raw_read import RawRead
# from misc.file_util import FileUtil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', nargs='+', default=['./data/newdata/KumonColor/ortho2b.raw', './data/newdata/KumonColor/ortho2a.raw'])
    parser.add_argument('--output', default='./output/extra/balochistan_deepflow')
    parser.add_argument('--cut_start', nargs='+', default=[2050, 1600], type=int)
    parser.add_argument('--cut_size', nargs='+', default=[1000, 1000], type=int)

    args = parser.parse_args()

    # FileUtil.mkdir(args.output)
    path = args.output
    if not os.path.exists(path):
        os.makedirs(path)

    cut_start = args.cut_start
    cut_size = args.cut_size

    images = []
    for path in args.image:
        images.append(RawRead.read(path, rate=1.)[cut_start[0]:cut_start[0] + cut_size[0], cut_start[1]:cut_start[1] + cut_size[1]])

    # 2枚の画像でdeepflow
    flow = np.empty_like(images[0])
    deepflow = cv2.optflow.createOptFlow_DeepFlow()
    flow = deepflow.calc(images[0], images[1], flow)
    print(flow.shape)

    # 画像にして出力
    for idx in range(2):
        cv2.imwrite(os.path.join(args.output, 'flow_' + str(idx) + '.png'), (flow[:, :, idx] * -100).astype('uint8'))
