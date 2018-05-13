# -*- coding: utf-8 -*-
from flask import render_template,flash, redirect, url_for,jsonify,request,Response,redirect
from . import app, emb, emb_torch
import json
import sys
import numpy as np
import cv2
from .embeddings import prewhiten
from .pytorch_net import test_transform
import torch
from torch.autograd import Variable

face_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def to_json(data):
    return json.dumps(data) + "\n"


def resp(code, data):
    return Response(
        status=code,
        mimetype="application/json",
        response=to_json(data)
    )


def img_validate():
    errors = []
    json = request.get_json()
    if json is None:
        errors.append(
            "No JSON sent. Did you forget to set Content-Type header" +
            " to application/json?")
        return (None, errors)
    return (json, errors)


@app.route('/facenet', methods=['POST','PUT'])
def facenet():
    data = prewhiten(np.frombuffer(request.data, dtype=np.float64).reshape(160,160,3))
    return resp(200, list(map(float, emb.extract(data[None])[0])))


@app.route('/manual', methods=['POST','PUT'])
def manual():
    data = test_transform(np.frombuffer(request.data, dtype=np.float64).reshape(218, 178, 3))
    var = Variable(torch.FloatTensor(data))
    emb = emb_torch.get_embeddings(var)[0]
    return resp(200, list(map(float, emb)))

