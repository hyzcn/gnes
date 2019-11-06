


from typing import List

import numpy as np

from ..base import BaseTextEncoder
from ...helper import batching, as_numpy_array


class CharEmbeddingEncoder(BaseTextEncoder):
    """A random character embedding model. Only useful for testing"""
    is_trained = True

    def __init__(self, dim: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.offset = 32
        self.unknown_idx = 96
        # in total 96 printable chars and 2 special chars = 98
        self._char_embedding = np.random.random([97, dim])

    @batching
    @as_numpy_array
    def encode(self, text: List[str], *args, **kwargs) -> List[np.ndarray]:
        # tokenize text
        sent_embed = []
        for sent in text:
            ids = [ord(c) - 32 if 32 <= ord(c) <= 127 else self.unknown_idx for c in sent]
            sent_embed.append(np.mean(self._char_embedding[ids], axis=0))
        return sent_embed
