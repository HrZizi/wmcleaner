from __future__ import annotations

from importlib import resources

import cv2
import numpy as np
from PIL import Image

from .detector import Detection


def _reference_rgb(size: tuple[int, int]) -> np.ndarray:
    asset = resources.files("wmcleaner").joinpath("templates", "baozimh_bottom_right.png")
    with resources.as_file(asset) as path:
        ref = Image.open(path).convert("RGB")
    if ref.size != size:
        ref = ref.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(ref, dtype=np.uint8)


def structural_overlap_score(image: Image.Image | np.ndarray, detection: Detection) -> float:
    """Estimate whether real dark comic structure passes through a logo overlay.

    The normal modern Baozimh watermark itself contains one dark dumpling glyph.
    We therefore compare the source page against the bundled watermark reference
    and count *additional* dark pixels only where the reference expects pale gray
    or yellow watermark foreground. This catches cases such as black speech-bubble
    text and narration-card borders passing through the watermark while avoiding
    ordinary watermark-only pages.
    """
    if detection.name != "baozimh_logo":
        return 0.0

    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image, dtype=np.uint8)

    x, y, w, h = detection.box
    roi = rgb[y:y + h, x:x + w]
    if roi.shape[:2] != (h, w) or roi.size == 0:
        return 0.0

    ref = _reference_rgb((w, h))
    src_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_RGB2HSV)

    reference_support = (
        ((255 - ref_gray) > 8)
        | (ref_hsv[:, :, 1] > 20)
    )

    # Ignore the watermark's own dark dumpling strokes. What matters is dark
    # source structure at locations where the watermark reference is pale/yellow.
    extra_dark = (
        (src_gray < 135)
        & (ref_gray > 165)
        & reference_support
    )

    denom = int(np.count_nonzero(reference_support))
    if denom == 0:
        return 0.0
    return float(np.count_nonzero(extra_dark) / denom)


def max_structural_overlap_score(
    image: Image.Image | np.ndarray,
    detections: list[Detection],
) -> float:
    return max((structural_overlap_score(image, d) for d in detections), default=0.0)
