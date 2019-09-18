# -*- coding: utf-8 -*-

"""
matching on correlation map
Author :
    Yuki Kumon
Last Update :
    2019-09-18
"""


try:
    from misc.Correlation_map import Correlation_map
    print('misc.Correlation_map loaded')
except ModuleNotFoundError as e:
    print(e)
try:
    from Correlation_map import Correlation_map
    print('Correlation_map loaded')
except ModuleNotFoundError as e:
    print(e)


class Matching():
    '''
    multi-level correlation pyramidからマッチングを行う
    '''

    def _init__(self, map):
        pass


if __name__ == '__main__':
    """
    sanity check
    """

    cls = Matching()
