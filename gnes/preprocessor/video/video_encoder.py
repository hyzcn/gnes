

from ...proto import gnes_pb2, blob2array
from ..base import BaseVideoPreprocessor
from ..io_utils import video, gif, webp


class VideoEncoderPreprocessor(BaseVideoPreprocessor):
    def __init__(self, frame_rate: int = 10, pix_fmt: str = 'rgb24', video_format: str = "mp4", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pix_fmt = pix_fmt
        self.frame_rate = frame_rate
        self.video_format = video_format

        if self.video_format not in ['mp4', 'gif', 'webp']:
            raise ValueError("%s encoder has not been supported!" % (self.video_format))

    def _encode(self, images: 'np.ndarray'):
        encoder = None
        if self.video_format == 'mp4':
            encoder = video
        elif self.video_format == 'gif':
            encoder = gif
        elif self.video_format == 'webp':
            encoder = webp

        return encoder.encode_video(images, pix_fmt=self.pix_fmt, frame_rate=self.frame_rate)

    def apply(self, doc: 'gnes_pb2.Document') -> None:
        super().apply(doc)
        if len(doc.chunks) > 0:
            for chunk in doc.chunks:
                images = blob2array(chunk.blob)
                chunk.raw = self._encode(images)
        elif doc.WhichOneof('raw_data'):
            raw_type = type(getattr(doc, doc.WhichOneof('raw_data')))
            if raw_type == gnes_pb2.NdArray:
                images = blob2array(doc.raw_video)
                doc.raw_bytes = self._encode(images)
            else:
                self.logger.error('bad document: "doc.raw_video" is empty!')
        else:
            self.logger.error(
                'bad document: "doc.chunks" and "doc.raw_video" is empty!')
