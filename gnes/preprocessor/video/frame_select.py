

import numpy as np
import math

from gnes.preprocessor.base import BaseVideoPreprocessor
from gnes.proto import gnes_pb2, array2blob, blob2array


class FrameSelectPreprocessor(BaseVideoPreprocessor):

    def __init__(self,
                 sframes: int = 1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sframes = sframes

    def apply(self, doc: 'gnes_pb2.Document') -> None:
        super().apply(doc)
        if len(doc.chunks) > 0:
            for chunk in doc.chunks:
                images = blob2array(chunk.blob)
                if len(images) == 0:
                    self.logger.warning("this chunk has no frame!")
                elif self.sframes == 1:
                    idx = [int(len(images) / 2)]
                    chunk.blob.CopyFrom(array2blob(images[idx]))
                elif self.sframes > 0 and len(images) > self.sframes:
                    if len(images) >= 2 * self.sframes:
                        step = math.ceil(len(images) / self.sframes)
                        chunk.blob.CopyFrom(array2blob(images[::step]))
                    else:
                        idx = np.sort(np.random.choice(len(images), self.sframes, replace=False))
                        chunk.blob.CopyFrom(array2blob(images[idx]))
                del images
        else:
            self.logger.error(
                'bad document: "doc.chunks" is empty!')
