


# A key-value map for Class to the (module)file it located in
from ..base import register_all_class

_cls2file_map = {
    'BasePreprocessor': 'base',
    'PipelinePreprocessor': 'base',
    'SentSplitPreprocessor': 'text.split',
    'BaseImagePreprocessor': 'base',
    'BaseTextPreprocessor': 'base',
    'VanillaSlidingPreprocessor': 'image.sliding_window',
    'WeightedSlidingPreprocessor': 'image.sliding_window',
    'SegmentPreprocessor': 'image.segmentation',
    'UnaryPreprocessor': 'base',
    'ResizeChunkPreprocessor': 'image.resize',
    'BaseVideoPreprocessor': 'base',
    'FFmpegPreprocessor': 'video.ffmpeg',
    'FFmpegVideoSegmentor': 'video.ffmpeg',
    'ShotDetectorPreprocessor': 'video.shot_detector',
    'VideoEncoderPreprocessor': 'video.video_encoder',
    'VideoDecoderPreprocessor': 'video.video_decoder',
    'AudioVanilla': 'audio.audio_vanilla',
    'BaseAudioPreprocessor': 'base',
    'RawChunkPreprocessor': 'base',
    'GifChunkPreprocessor': 'video.ffmpeg',
    'VggishPreprocessor': 'audio.vggish_example',
    'FrameSelectPreprocessor': 'video.frame_select'
}

register_all_class(_cls2file_map, 'preprocessor')
