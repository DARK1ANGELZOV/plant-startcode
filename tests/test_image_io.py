from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from utils.image_io import decode_image_bytes


def test_decode_image_bytes_handles_empty_payload() -> None:
    assert decode_image_bytes(b'') is None


def test_decode_image_bytes_decodes_png() -> None:
    src = np.zeros((12, 18, 3), dtype=np.uint8)
    src[:, :] = (10, 120, 250)  # RGB
    image = Image.fromarray(src, mode='RGB')
    buf = BytesIO()
    image.save(buf, format='PNG')

    out = decode_image_bytes(buf.getvalue())
    assert out is not None
    assert out.shape[:2] == (12, 18)
    # Converted to BGR
    b, g, r = out[0, 0].tolist()
    assert int(r) == 10
    assert int(g) == 120
    assert int(b) == 250


def test_decode_image_bytes_respects_exif_orientation() -> None:
    src = np.zeros((10, 20, 3), dtype=np.uint8)
    src[:, :10] = (255, 0, 0)  # red half in RGB
    image = Image.fromarray(src, mode='RGB')
    exif = image.getexif()
    exif[274] = 6  # rotate 90 CW

    buf = BytesIO()
    image.save(buf, format='JPEG', exif=exif)

    out = decode_image_bytes(buf.getvalue())
    assert out is not None
    # EXIF transpose should rotate from (10,20) to (20,10)
    assert out.shape[:2] == (20, 10)
