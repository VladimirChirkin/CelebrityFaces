#!/usr/bin/env python
import os

class Config(object):
    debug = True
    testing = False
    UPLOAD_FOLDER="celebrity_faces/temp"
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'b09c6c49-e43c-4163-a5f1-6c2fdd9ffa06'

class ManualConfig():
    host_emb='0.0.0.0'
    port_emb=5001

    host_index = '0.0.0.0'
    port_index = 5002

    net_type='manual'
    net_types_shape={'facenet':(160,160),'manual':(218,178)}