


import numpy as np

from ..base import BaseNumericEncoder
from ...helper import batching, train_required


class StandarderEncoder(BaseNumericEncoder):
    batch_size = 2048

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = None
        self.scale = None

    def post_init(self):
        from sklearn.preprocessing import StandardScaler
        self.standarder = StandardScaler()

    @batching
    def train(self, vecs: np.ndarray, *args, **kwargs) -> None:
        self.standarder.partial_fit(vecs)

        self.mean = self.standarder.mean_.astype('float32')
        self.scale = self.standarder.scale_.astype('float32')

    @train_required
    @batching
    def encode(self, vecs: np.ndarray, *args, **kwargs) -> np.ndarray:
        return (vecs - self.mean) / self.scale