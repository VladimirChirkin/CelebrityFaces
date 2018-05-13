# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

"""
These functions extract embeddings using pretrained facenet model in tensorflow.
Use prewhiten to normalize images.
You need facenet files to initialize TfExtractor.
Input shape is [batch, height, width, channels]
"""


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


class TfExtractor:
    def __init__(self, meta_graph_addr='models/model-20180402-114759.meta',
                 model_addr='models/model-20180402-114759.ckpt-275.data-00000-of-00001'):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(meta_graph_addr)
        saver.restore(self.sess, model_addr)
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    def extract(self, batch):
        feed_dict = {self.images_placeholder: batch, self.phase_train_placeholder: False}
        return self.sess.run(self.embeddings, feed_dict=feed_dict)
