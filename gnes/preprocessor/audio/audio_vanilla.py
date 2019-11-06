

import numpy as np

from ..base import BaseAudioPreprocessor
from ...proto import array2blob
from ..io_utils.audio import split_audio


class AudioVanilla(BaseAudioPreprocessor):

    def __init__(self,
                 sample_rate: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_rate = sample_rate

    def apply(self, doc: 'gnes_pb2.Document') -> None:
        super().apply(doc)

        if doc.raw_bytes:
            audio = split_audio(input_data=doc.raw_bytes, sample_rate=self.sample_rate)
            if len(audio) >= 1:
                for ci, chunks in enumerate(audio):
                    c = doc.chunks.add()
                    c.doc_id = doc.doc_id
                    c.blob.CopyFrom(array2blob(np.array(chunks, dtype=np.float32)))
                    c.offset = ci
                    c.weight = 1 / len(audio)
            else:
                self.logger.warning('bad document: no audio extracted')
        else:
            self.logger.error('bad document: "raw_bytes" is empty!')
