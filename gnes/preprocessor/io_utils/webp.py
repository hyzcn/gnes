

from typing import List

def encode_video(images: List['np.ndarray'], frame_rate: int, pix_fmt: str = 'rgb24'):
    import webp
    from PIL import Image

    height, width, _ = images[0].shape
    if pix_fmt == 'rgb24':
        pix_fmt = 'RGB'
    # Save an animation
    enc = webp.WebPAnimEncoder.new(width, height)
    timestamp_ms = 0
    duration = 1000 // frame_rate
    for x in images:
        img = Image.fromarray(x.copy(), pix_fmt)
        pic = webp.WebPPicture.from_pil(img)
        enc.encode_frame(pic, timestamp_ms)
        timestamp_ms += duration
    anim_data = enc.assemble(timestamp_ms)
    return bytes(anim_data.buffer())
