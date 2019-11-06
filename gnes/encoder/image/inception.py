

from typing import List

import numpy as np

from ..base import BaseImageEncoder
from ...helper import batching, get_first_available_gpu


class TFInceptionEncoder(BaseImageEncoder):
    batch_size = 64

    def __init__(self, model_dir: str,
                 select_layer: str = 'PreLogitsFlatten',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_dir = model_dir
        self.select_layer = select_layer
        self.inception_size_x = 299
        self.inception_size_y = 299

    def post_init(self):
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(get_first_available_gpu())

        import tensorflow as tf
        from .inception_cores.inception_v4 import inception_v4
        from .inception_cores.inception_utils import inception_arg_scope
        g = tf.Graph()
        with g.as_default():
            arg_scope = inception_arg_scope()
            inception_v4.default_image_size = self.inception_size_x
            self.inputs = tf.placeholder(tf.float32, (None,
                                                      self.inception_size_x,
                                                      self.inception_size_y, 3))

            with tf.contrib.slim.arg_scope(arg_scope):
                self.logits, self.end_points = inception_v4(self.inputs,
                                                            is_training=False,
                                                            dropout_keep_prob=1.0)

            config = tf.ConfigProto(log_device_placement=False)
            if self.on_gpu:
                config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.model_dir)

    def encode(self, img: List['np.ndarray'], *args, **kwargs) -> np.ndarray:
        img = [(im * 2 / 255. - 1.) for im in img]

        @batching
        def _encode(_, data):
            _, end_points_ = self.sess.run((self.logits, self.end_points),
                                           feed_dict={self.inputs: data})
            return end_points_[self.select_layer]

        return _encode(self, img).astype(np.float32)
