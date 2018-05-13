# -*- coding: utf-8 -*-
from flask import render_template,flash, redirect, url_for,jsonify,request,Response,redirect
from . import app, ind
from .config import ManualConfig
import json
import sys
import numpy as np

def to_json(data):
    return json.dumps(data) + "\n"

def resp(code, data):
    return Response(
        status=code,
        mimetype="application/json",
        response=to_json(data)
    )

@app.route('/',methods=['POST','PUT'])
def root():
    conf = ManualConfig()
    vector = np.array(json.loads(request.data.decode()),float)
    return resp(200, list(map(int,ind.get_most_similar(vector,conf.n_similar))))

