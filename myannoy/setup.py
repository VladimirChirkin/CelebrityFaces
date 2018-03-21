#!/usr/bin/python3
from distutils.core import setup
from Cython.Build import cythonize

setup(name='Annoy', ext_modules=cythonize('annoyforest.pyx'))
