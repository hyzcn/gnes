


from typing import List

import numpy as np

from ..base import BaseTextEncoder
from ...helper import batching, pooling_simple, as_numpy_array


class Word2VecEncoder(BaseTextEncoder):
    is_trained = True

    def __init__(self, model_dir: str,
                 skiprows: int = 1,
                 dimension: int = 300,
                 pooling_strategy: str = 'REDUCE_MEAN', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dir = model_dir
        self.skiprows = skiprows
        self.pooling_strategy = pooling_strategy
        self.dimension = dimension

    def post_init(self):
        from ...helper import Tokenizer
        count = 0
        self.word2vec_df = {}
        with open(self.model_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                if count < self.skiprows:
                    count += 1
                    continue
                if len(line) > self.dimension:
                    self.word2vec_df[line[0]] = np.array([float(i) for i in line[1:]], dtype=np.float32)

        self.empty = np.zeros([self.dimension], dtype=np.float32)
        self.cn_tokenizer = Tokenizer()

    @batching
    @as_numpy_array
    def encode(self, text: List[str], *args, **kwargs) -> np.ndarray:
        # tokenize text
        batch_tokens = [self.cn_tokenizer.tokenize(sent) for sent in text]
        pooled_data = []

        for tokens in batch_tokens:
            _layer_data = [self.word2vec_df.get(token, self.empty) for token in tokens]
            pooled_data.append(pooling_simple(_layer_data, self.pooling_strategy))

        return pooled_data
