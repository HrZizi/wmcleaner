from __future__ import annotations

import unittest
from importlib import resources

import numpy as np
from PIL import Image

from wmcleaner.comic_overlap import structural_overlap_score
from wmcleaner.detector import Detection


class ComicOverlapTests(unittest.TestCase):
    def _case(self, crossing: bool) -> tuple[Image.Image, Detection]:
        with resources.as_file(
            resources.files("wmcleaner").joinpath("templates", "baozimh_bottom_right.png")
        ) as p:
            ref = Image.open(p).convert("RGB")
        page = np.asarray(ref, dtype=np.uint8).copy()
        mask = np.full(page.shape[:2], 255, dtype=np.uint8)
        if crossing:
            page[55:61, 105:280] = 0
            page[70:76, 125:285] = 0
        det = Detection("baozimh_logo", 0.95, (0, 0, ref.width, ref.height), mask)
        return Image.fromarray(page), det

    def test_plain_logo_does_not_trigger_structural_overlap(self) -> None:
        image, det = self._case(False)
        self.assertLess(structural_overlap_score(image, det), 0.01)

    def test_dark_comic_structure_through_pale_watermark_triggers(self) -> None:
        image, det = self._case(True)
        self.assertGreater(structural_overlap_score(image, det), 0.035)

    def test_non_logo_detection_scores_zero(self) -> None:
        image, det = self._case(True)
        other = Detection("baozimh_top_left", det.confidence, det.box, det.mask)
        self.assertEqual(structural_overlap_score(image, other), 0.0)


if __name__ == "__main__":
    unittest.main()
