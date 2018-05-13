#!/usr/bin/env python
import os

class Config(object):
    #SERVER_NAME = "0.0.0.0:5001"
    debug = True
    testing = True
    UPLOAD_FOLDER="~"
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'b09c6c49-e43c-4163-a5f1-6c2fdd9ffa06'
