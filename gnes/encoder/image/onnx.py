

from typing import List

import numpy as np

from ..base import BaseImageEncoder
from ...helper import batching


class BaseONNXImageEncoder(BaseImageEncoder):
    batch_size = 64

    def __init__(self, model_name: str,
                 model_dir: str,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_dir = model_dir
        self.model_name = model_name

    def post_init(self):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(self.model_dir + '/' + self.model_name)
        inputs_info = self.sess.get_inputs()

        if len(inputs_info) != 1:
            raise ValueError('Now only support encoder with one input')
        else:
            self.input_name = inputs_info[0].name
            self.input_shape = inputs_info[0].shape
            self.input_type = inputs_info[0].type
            self.batch_size = self.input_shape[0]

    @batching
    def encode(self, img: List['np.ndarray'], *args, **kwargs) -> np.ndarray:
        pad_batch = 0
        if len(img) != self.input_shape[0]:
            pad_batch = self.input_shape[0] - len(img)
            for _ in range(pad_batch):
                img.append(np.zeros_like(img[0]))

        img_ = np.array(img, dtype=np.float32).transpose(0, 3, 1, 2)
        if list(img_.shape) != self.input_shape:
            raise ValueError('Map size not match net, expect', self.input_shape, ',got', img_.shape)

        result_npy = self.sess.run(None, {self.input_name: img_})
        if pad_batch != 0:
            return result_npy[0][0:len(img)]
        else:
            return result_npy[0]
