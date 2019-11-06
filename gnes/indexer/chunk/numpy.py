


from typing import List, Tuple, Any

import numpy as np

from .helper import ListKeyIndexer
from ..base import BaseChunkIndexer as BCI


class NumpyIndexer(BCI):
    """An exhaustive search indexer using numpy
    The distance is computed as L1 distance normalized by the number of dimension
    """

    def __init__(self, is_binary: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_dim = None
        self._vectors = None  # type: np.ndarray
        self._is_binary = is_binary
        self.helper_indexer = ListKeyIndexer()

    @BCI.update_helper_indexer
    def add(self, keys: List[Tuple[int, Any]], vectors: np.ndarray, weights: List[float], *args,
            **kwargs):
        if len(vectors) % len(keys) != 0:
            raise ValueError('vectors bytes should be divided by doc_ids')

        if not self._num_dim:
            self._num_dim = vectors.shape[1]
        elif self._num_dim != vectors.shape[1]:
            raise ValueError(
                "vectors' shape [%d, %d] does not match with indexer's dim: %d" %
                (vectors.shape[0], vectors.shape[1], self._num_dim))

        if self._vectors is not None:
            self._vectors = np.concatenate([self._vectors, vectors], axis=0)
        else:
            self._vectors = vectors

    def query(self, keys: np.ndarray, top_k: int, *args, **kwargs) -> List[List[Tuple]]:
        dist = np.abs(np.expand_dims(keys, axis=1) - np.expand_dims(self._vectors, axis=0))

        if self._is_binary:
            dist = np.minimum(dist, 1)

        score = np.sum(dist, -1) / self._num_dim

        ret = []
        for ids in score:
            rk = sorted(enumerate(ids), key=lambda x: x[1])[:top_k]
            chunk_info = self.helper_indexer.query([j[0] for j in rk])
            ret.append([(*r, s) for r, s in zip(chunk_info, [j[1] for j in rk])])
        return ret
