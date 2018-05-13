#!/usr/bin/env python
import torch
from flask import Flask
from .config import Config
from .embeddings import TfExtractor
from .pytorch_net import MyNet
   
app = Flask(__name__)
app.config.from_object(Config)
emb=TfExtractor(meta_graph_addr='get_embeddings/models/model-20170512-110547.meta',
                 model_addr='get_embeddings/models/model-20170512-110547.ckpt-250000')

emb_torch=MyNet()
emb_torch.load_state_dict(torch.load('get_embeddings/models/model_classif'))

from . import routes