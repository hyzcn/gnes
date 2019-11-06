

import numpy as np
from typing import List
from ..base import BaseAudioEncoder
from ...helper import batching, get_first_available_gpu


class VggishEncoder(BaseAudioEncoder):
    def __init__(self, model_dir: str,
                 max_length: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dir = model_dir
        self.max_length = max_length

    def post_init(self):
        import os
        import tensorflow as tf
        from .vggish_cores import vggish_slim
        from .vggish_cores import vggish_params
        from .vggish_cores import vggish_postprocess

        os.environ['CUDA_VISIBLE_DEVICES'] = str(get_first_available_gpu())

        self.graph = tf.Graph()

        with tf.Graph().as_default():
            self._sess = tf.Session()
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self._sess, self.model_dir + "/vggish_model.ckpt")
            self._audio_inputs = self._sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            self._embedding = self._sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        self._pproc = vggish_postprocess.Postprocessor(self.model_dir + "/vggish_pca_params.npz")


    @batching
    def encode(self, audio: List['np.ndarray'], *args, **kwargs) -> np.ndarray:
        # used for np.split()
        audio_length = [len(au) for au in audio]
        for i in range(1, len(audio_length)):
            audio_length[i] = audio_length[i] + audio_length[i - 1]

        # concat at axis = 0 to make sure the dimension is: (#vggish_example, D)
        audio_ = np.concatenate((list(audio[i] for i in range(len(audio)))), axis=0)

        [embedding_batch] = self._sess.run([self._embedding], feed_dict={self._audio_inputs: audio_})
        postprocessed_batch = self._pproc.postprocess(embedding_batch)

        audio_features = np.split(postprocessed_batch, audio_length[:-1])
        return np.array([features.mean(axis=0) for features in audio_features])