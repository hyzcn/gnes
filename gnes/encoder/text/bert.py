


from typing import List

import numpy as np

from ..base import CompositionalTrainableBase, BaseTextEncoder
from ...helper import batching


class BertEncoder(BaseTextEncoder):
    store_args_kwargs = True
    is_trained = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bc_encoder_args = args
        self._bc_encoder_kwargs = kwargs

    def post_init(self):
        from bert_serving.client import BertClient
        self.bc_encoder = BertClient(*self._bc_encoder_args, **self._bc_encoder_kwargs)

    @batching
    def encode(self, text: List[str], *args, **kwargs) -> np.ndarray:
        return self.bc_encoder.encode(text, *args, **kwargs)  # type: np.ndarray

    def close(self):
        self.bc_encoder.close()


class BertEncoderWithServer(CompositionalTrainableBase):
    def encode(self, text: List[str], *args, **kwargs) -> np.ndarray:
        return self.components['bert_client'].encode(text, *args, **kwargs)


class BertEncoderServer(BaseTextEncoder):
    store_args_kwargs = True
    is_trained = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bert_args = ['-%s' % v for v in args]
        for k, v in kwargs.items():
            bert_args.append('-%s' % k)
            bert_args.append(str(v))
        self._bert_args = bert_args

    def post_init(self):
        from bert_serving.server import BertServer
        from bert_serving.server import get_args_parser
        self.bert_server = BertServer(get_args_parser().parse_args(self._bert_args))
        self.bert_server.start()
        self.bert_server.is_ready.wait()

    def close(self):
        self.bert_server.close()
