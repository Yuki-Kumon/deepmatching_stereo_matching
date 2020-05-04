#! -*- coding: utf-8 -*-

"""
バイラテラルフィルターの練習用のプログラムです。
Usage :
    $ python setup.py build_ext --inplace
Author :
    Yuki Kumon
Last Update :
    2018-06-24
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension("Filter_optimize", ["Script.pyx"]),
]

setup(
    name = "Filter App",
    cmdclass = {"build_ext" : build_ext },
    ext_modules = ext_modules,
    include_dirs = [numpy.get_include()]
)
