


import os
from typing import List, Tuple, Any

import numpy as np

from .helper import ListKeyIndexer
from ..base import BaseChunkIndexer as BCI


class FaissIndexer(BCI):

    def __init__(self, num_dim: int, index_key: str, data_path: str, *args, **kwargs):
        """
        Initialize an FaissIndexer

        :param num_dim: when set to -1, then num_dim is auto decided on first .add()
        :param data_path: index data file managed by the faiss indexer
        """
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.num_dim = num_dim
        self.index_key = index_key
        self.helper_indexer = ListKeyIndexer()

    def post_init(self):
        import faiss
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError('"data_path" is not exist')
            if os.path.isdir(self.data_path):
                raise IsADirectoryError('"data_path" must be a file path, not a directory')
            self._faiss_index = faiss.read_index(self.data_path)
        except (RuntimeError, FileNotFoundError, IsADirectoryError):
            self.logger.warning('fail to load model from %s, will init an empty one' % self.data_path)
            self._faiss_index = faiss.index_factory(self.num_dim, self.index_key) if self.num_dim > 0 else None

    @BCI.update_helper_indexer
    def add(self, keys: List[Tuple[int, Any]], vectors: np.ndarray, weights: List[float], *args, **kwargs):
        if len(vectors) != len(keys):
            raise ValueError('vectors length should be equal to doc_ids')

        if vectors.dtype != np.float32:
            raise ValueError('vectors should be ndarray of float32')

        if self._faiss_index is None:
            import faiss
            # means num_dim in unknown during init
            self.num_dim = vectors.shape[1]
            self._faiss_index = faiss.index_factory(self.num_dim, self.index_key)

        self._faiss_index.add(vectors)

    def query(self, keys: np.ndarray, top_k: int, *args, **kwargs) -> List[List[Tuple]]:
        if keys.dtype != np.float32:
            raise ValueError('vectors should be ndarray of float32')

        score, ids = self._faiss_index.search(keys, top_k)
        ret = []
        for _id, _score in zip(ids, score):
            ret_i = []
            chunk_info = self.helper_indexer.query(_id)
            for c_info, _score_i in zip(chunk_info, _score):
                ret_i.append((*c_info, _score_i))
            ret.append(ret_i)

        return ret

    def __getstate__(self):
        import faiss
        d = super().__getstate__()
        faiss.write_index(self._faiss_index, self.data_path)
        return d
