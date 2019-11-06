


from typing import List, Any, Tuple, Union

import numpy as np

from ..base import TrainableBase, CompositionalTrainableBase


class BaseEncoder(TrainableBase):

    def encode(self, data: Any, *args, **kwargs) -> Any:
        pass

    def _copy_from(self, x: 'BaseEncoder') -> None:
        pass


class BaseImageEncoder(BaseEncoder):

    def encode(self, img: List['np.ndarray'], *args, **kwargs) -> np.ndarray:
        pass


class BaseVideoEncoder(BaseEncoder):

    def encode(self, data: List['np.ndarray'], *args, **kwargs) -> Union[np.ndarray, List['np.ndarray']]:
        pass


class BaseTextEncoder(BaseEncoder):

    def encode(self, text: List[str], *args, **kwargs) -> Union[Tuple, np.ndarray]:
        pass


class BaseNumericEncoder(BaseEncoder):
    """Note that all NumericEncoder can not be used as the first encoder of the pipeline"""

    def encode(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass


class BaseAudioEncoder(BaseEncoder):

    def encode(self, data: List['np.ndarray'], *args, **kwargs) -> np.ndarray:
        pass


class BaseBinaryEncoder(BaseEncoder):

    def encode(self, data: np.ndarray, *args, **kwargs) -> bytes:
        if data.dtype != np.uint8:
            raise ValueError('data must be np.uint8 but received %s' % data.dtype)
        return data.tobytes()


class PipelineEncoder(CompositionalTrainableBase):
    def encode(self, data: Any, *args, **kwargs) -> Any:
        if not self.components:
            raise NotImplementedError
        for be in self.components:
            data = be.encode(data, *args, **kwargs)
        return data

    def train(self, data, *args, **kwargs):
        if not self.components:
            raise NotImplementedError
        for idx, be in enumerate(self.components):
            if not be.is_trained:
                be.train(data, *args, **kwargs)

            if idx + 1 < len(self.components):
                data = be.encode(data, *args, **kwargs)
