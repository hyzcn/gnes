

from typing import List

import numpy as np
from PIL import Image

from ..base import BaseImageEncoder


class CVAEEncoder(BaseImageEncoder):
    batch_size = 64

    def __init__(self, model_dir: str,
                 latent_dim: int = 300,
                 select_method: str = 'MEAN',
                 l2_normalize: bool = False,
                 use_gpu: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_dir = model_dir
        self.latent_dim = latent_dim
        self.select_method = select_method
        self.l2_normalize = l2_normalize
        self.use_gpu = use_gpu

    def post_init(self):
        import tensorflow as tf
        from .cvae_cores.model import CVAE
        g = tf.Graph()
        with g.as_default():
            self._model = CVAE(self.latent_dim)
            self.inputs = tf.placeholder(tf.float32,
                                         (None, 120, 120, 3))

            self.mean, self.var = self._model.encode(self.inputs)

            config = tf.ConfigProto(log_device_placement=False)
            if self.use_gpu:
                config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.model_dir)

    def encode(self, img: List['np.ndarray'], *args, **kwargs) -> np.ndarray:
        img = [(np.array(Image.fromarray(im).resize((120, 120)),
                         dtype=np.float32) / 255) for im in img]

        def _encode(_, data):
            _mean, _var = self.sess.run((self.mean, self.var),
                                        feed_dict={self.inputs: data})
            if self.select_method == 'MEAN':
                return _mean
            elif self.select_method == 'VAR':
                return _var
            elif self.select_method == 'MEAN_VAR':
                return np.concatenate([_mean, _var], axis=1)
            else:
                raise NotImplementedError

        v = _encode(None, img).astype(np.float32)
        if self.l2_normalize:
            v = v / (v ** 2).sum(axis=1, keepdims=True) ** 0.5
        return v
