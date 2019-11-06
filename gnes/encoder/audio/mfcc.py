


from typing import List

import numpy as np

from ..base import BaseAudioEncoder
from ...helper import batching


class MfccEncoder(BaseAudioEncoder):
    batch_size = 64

    def __init__(self, n_mfcc: int = 13, sample_rate: int = 16000, max_length: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.max_length = max_length

    @batching
    def encode(self, data: List['np.array'], *args, **kwargs) -> np.ndarray:
        import librosa

        mfccs = [np.array(librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc).T)
                 for audio in data]

        mfccs = [np.concatenate((mf, np.zeros((self.max_length - mf.shape[0], self.n_mfcc), dtype=np.float32)), axis=0)
                 if mf.shape[0] < self.max_length else mf[:self.max_length] for mf in mfccs]
        mfccs = [mfcc.reshape((1, -1)) for mfcc in mfccs]
        mfccs = np.squeeze(np.array(mfccs), axis=1)
        return mfccs
