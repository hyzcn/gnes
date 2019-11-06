


from typing import List, Tuple

import numpy as np

from ..base import BaseTextEncoder
from ...helper import batching, as_numpy_array


class FlairEncoder(BaseTextEncoder):
    is_trained = True

    def __init__(self,
                 word_embedding: str = 'glove',
                 flair_embeddings: Tuple[str] = ('news-forward', 'news-backward'),
                 pooling_strategy: str = 'mean', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.word_embedding = word_embedding
        self.flair_embeddings = flair_embeddings
        self.pooling_strategy = pooling_strategy

    def post_init(self):
        from flair.embeddings import DocumentPoolEmbeddings, WordEmbeddings, FlairEmbeddings
        self._flair = DocumentPoolEmbeddings(
            [WordEmbeddings(self.word_embedding),
             FlairEmbeddings(self.flair_embeddings[0]),
             FlairEmbeddings(self.flair_embeddings[1])],
            pooling=self.pooling_strategy)

    @batching
    @as_numpy_array
    def encode(self, text: List[str], *args, **kwargs) -> np.ndarray:
        from flair.data import Sentence
        import torch
        # tokenize text
        batch_tokens = [Sentence(v) for v in text]
        self._flair.embed(batch_tokens)
        return torch.stack([v.embedding for v in batch_tokens]).detach()
