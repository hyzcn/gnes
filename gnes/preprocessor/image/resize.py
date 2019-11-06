

import numpy as np
from PIL import Image

from ..base import BaseImagePreprocessor
from ...proto import gnes_pb2, blob2array, array2blob


class SizedPreprocessor(BaseImagePreprocessor):
    def __init__(self,
                 target_width: int = 224,
                 target_height: int = 224,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_width = target_width
        self.target_height = target_height


class ResizeChunkPreprocessor(SizedPreprocessor):

    def apply(self, doc: 'gnes_pb2.Document') -> None:
        super().apply(doc)
        for c in doc.chunks:
            img = blob2array(c.blob)
            img = np.array(Image.fromarray(img.astype('uint8')).resize((self.target_width, self.target_height)))
            c.blob.CopyFrom(array2blob(img))
