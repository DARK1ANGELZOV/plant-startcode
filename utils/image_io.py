from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps


def decode_image_bytes(image_bytes: bytes) -> np.ndarray | None:
    """Decode image bytes into BGR image with EXIF orientation handling."""
    if not image_bytes:
        return None

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            oriented = ImageOps.exif_transpose(img)
            rgb = oriented.convert('RGB')
            arr = np.asarray(rgb, dtype=np.uint8)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if bgr.size > 0:
                return bgr
    except Exception:
        pass

    raw = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if raw is None or raw.size == 0:
        return None
    return raw
