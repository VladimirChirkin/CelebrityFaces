# -*- coding: utf-8 -*-
import pickle
from .lib.annoyforest import Annoy

from .config import ManualConfig

class Index():
    def __init__(self,embs_path='index/embeddings'):
        self.conf = ManualConfig()
        embs = pickle.load(open(embs_path, 'rb'))
        embs = embs.astype(float)
        model = Annoy(self.conf.max_node_size, self.conf.n_trees)
        model.fit(embs)
        self.model=model

    def get_most_similar(self,vector,n_neighbors):
        return self.model.find(vector, self.conf.n_search)[:n_neighbors]


