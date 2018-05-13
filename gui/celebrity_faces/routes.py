#!/usr/bin/env python

import cv2
import http.client
import os
import io
import base64
import json

from flask import Flask, render_template, flash, request
from wtforms import Form, TextAreaField, validators, StringField, SubmitField
import numpy as np

from . import app
from .config import ManualConfig
from .crop import crop_face

all_photos=sorted(os.listdir('img_align_celeba'))
conf = ManualConfig()

def allowed_file(filename):
    allowed_ext = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_ext

class ReusableForm(Form):
    name = StringField('Name:', validators=[validators.required()])

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not os.path.isdir(app.config['UPLOAD_FOLDER']):
            os.mkdir(app.config['UPLOAD_FOLDER'])
        for file in request.files.getlist("file"):
            filename = file.filename
            if allowed_file(filename):
                nparr = np.fromfile(file, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                croped_img = crop_face(img_np, conf.net_types_shape[conf.net_type])

                vector=get_embeddings(croped_img.tobytes())
                similar_indexes=get_indexes(vector)
            file.seek(0)
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
        params={'origin_image' : base64.b64encode(in_memory_file.getvalue()).decode(),
                'cropped_image' : base64.b64encode(cv2.imencode('.jpeg', croped_img)[1].tobytes()).decode(),
                'is_visible': True}
        candidates=[]
        for ind, i in enumerate(similar_indexes):
            with open(os.path.join('img_align_celeba',all_photos[i]),'rb') as cand:
                candidates.append({
                  "id": "Candidate {}".format(ind+1),
                  "image":base64.b64encode(cand.read()).decode()
                })
        return render_template("index.html",candidates=candidates,**params)
    return render_template('index.html',is_visible=False)

def get_embeddings(file):
    conn = http.client.HTTPConnection("{}:{}".format(conf.host_emb, conf.port_emb))
    conn.request("PUT", "/{}".format(conf.net_type), file )
    response = conn.getresponse()
    if response.status==200:
        return json.loads(response.read().decode())

def get_indexes(vector):
    conn = http.client.HTTPConnection("{}:{}".format(conf.host_index, conf.port_index))
    conn.request("PUT", "/", json.dumps(vector)+ "\n")

    response = conn.getresponse()
    if response.status == 200:
        return json.loads(response.read().decode())
