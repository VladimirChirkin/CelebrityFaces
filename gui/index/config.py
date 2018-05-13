#!/usr/bin/env python
import os

class Config(object):
    debug = True
    testing = True
    UPLOAD_FOLDER="~"
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'b09c6c49-e43c-4163-a5f1-6c2fdd9ffa06'

class ManualConfig():
    n_trees=8
    max_node_size=80
    n_search=650
    n_similar=6
    net_type='manual' # "facenet" or "manual"
