# -*- coding: utf-8 -*-

"""
read raw data
Author :
    Yuki Kumon
Last Update :
    2020-01-17
"""


import numpy as np


class RawRead():
    '''
    read raw image and convert it to numpy array
    '''

    def __init__(self):
        pass

    @staticmethod
    def _read8(filename, xdata, ydata, band):
        """
        read hyperspectral data
        c = readHyper16(filename,xdata,ydata,bandnumber)
        Input
        filename: file name path
        xdata: cross line size
        ydata: along line size
        bandnumber: number of band
        Output
        c: array of data (c[band,y,x])
        """
        c = np.fromfile(filename, dtype=np.int8, count=xdata * ydata * band).reshape(band, ydata, xdata)
        # c = c.transpose(1, 2, 0)
        return c

    @classmethod
    def read(self, path, size=(6000, 6000), rate=1):
        '''
        read image
        '''
        return (self._read8(path, size[0], size[1], 1) * rate)[0].astype(np.uint8)


if __name__ == '__main__':
    """
    sanity check
    """

    import sys
    sys.path.append('.')
    from misc.image_cut_solver import ImageCutSolver
    path = './data/newdata/orthoa.raw'
    arr = RawRead.read(path)
    # print(np.max(arr[0]))
    ImageCutSolver.image_save('./output/test.png', (arr[0]).astype(np.uint8), threshold=[0, 255])
