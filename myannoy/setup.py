#!/usr/bin/python3
from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include

setup(name='Annoy', ext_modules=cythonize('annoyforest.pyx'), include_dirs=['.', get_include()])
