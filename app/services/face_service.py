from __future__ import annotations

from PIL import Image
from facenet_pytorch import MTCNN

_detector: MTCNN | None = None


def load_detector() -> None:
    global _detector
    _detector = MTCNN(keep_all=True)


def is_human_face(image: Image.Image) -> bool:
    try:
        if _detector is None:
            return False
        boxes, _ = _detector.detect(image)
        return boxes is not None and len(boxes) > 0
    except Exception:
        return False
