

import os
from typing import List, Tuple, Any

import numpy as np

from .helper import ListKeyIndexer
from ..base import BaseChunkIndexer as BCI


class AnnoyIndexer(BCI):

    def __init__(self, num_dim: int, data_path: str, metric: str = 'angular', n_trees: int = 10, *args, **kwargs):
        """
        Initialize an AnnoyIndexer

        :param num_dim: when set to -1, then num_dim is auto decided on first .add()
        :param data_path: index data file managed by the annoy indexer
        :param metric:
        :param n_trees:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.num_dim = num_dim
        self.data_path = data_path
        self.metric = metric
        self.n_trees = n_trees
        self.helper_indexer = ListKeyIndexer()

    def post_init(self):
        from annoy import AnnoyIndex
        self._index = AnnoyIndex(self.num_dim, self.metric) if self.num_dim >= 0 else None
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError('"data_path" is not exist')
            if os.path.isdir(self.data_path):
                raise IsADirectoryError('"data_path" must be a file path, not a directory')
            self._index.load(self.data_path)
        except:
            self.logger.warning('fail to load model from %s, will create an empty one' % self.data_path)

    @BCI.update_helper_indexer
    def add(self, keys: List[Tuple[int, Any]], vectors: np.ndarray, weights: List[float], *args, **kwargs):
        last_idx = self.helper_indexer.num_chunks

        if len(vectors) != len(keys):
            raise ValueError('vectors length should be equal to doc_ids')

        if vectors.dtype != np.float32:
            raise ValueError("vectors should be ndarray of float32")

        if self._index is None:
            from annoy import AnnoyIndex
            # means num_dim in unknown during init
            self.num_dim = vectors.shape[1]
            self._index = AnnoyIndex(self.num_dim, self.metric)

        for idx, vec in enumerate(vectors):
            self._index.add_item(last_idx + idx, vec)

    def query(self, keys: 'np.ndarray', top_k: int, *args, **kwargs) -> List[List[Tuple]]:
        self._index.build(self.n_trees)
        if keys.dtype != np.float32:
            raise ValueError('vectors should be ndarray of float32')
        res = []
        for k in keys:
            ret, relevance_score = self._index.get_nns_by_vector(k, top_k, include_distances=True)
            chunk_info = self.helper_indexer.query(ret)
            res.append([(*r, s) for r, s in zip(chunk_info, relevance_score)])
        return res

    def __getstate__(self):
        d = super().__getstate__()
        self._index.save(self.data_path)
        return d
