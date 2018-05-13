#!/usr/bin/env python
from flask import Flask
from .config import Config,ManualConfig
from .get_index import Index
   
app = Flask(__name__)
app.config.from_object(Config)
ind=Index('index/embeddings_{}'.format(ManualConfig.net_type))

from . import routes