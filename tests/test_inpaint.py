from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from wmcleaner.inpaint import Inpainter, _fit_backend_result


class _PaddedLama:
    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        # Mimic LaMa's modulo padding: source width/height grow only on the
        # bottom/right. Fill the model output with red so masked pixels are easy
        # to distinguish from the source in the compositing assertion below.
        w, h = image.size
        padded_w = ((w + 7) // 8) * 8
        padded_h = ((h + 7) // 8) * 8
        return Image.new("RGB", (padded_w, padded_h), (255, 0, 0))


class InpaintSizingTests(unittest.TestCase):
    def test_fit_backend_result_crops_bottom_right_padding(self) -> None:
        padded = Image.new("RGB", (856, 1208), (10, 20, 30))
        fitted = _fit_backend_result(padded, (850, 1205))
        self.assertEqual(fitted.size, (850, 1205))

    def test_lama_padded_output_keeps_original_size_and_unmasked_pixels(self) -> None:
        source = Image.new("RGB", (850, 1205), (20, 40, 60))
        mask_array = np.zeros((1205, 850), dtype=np.uint8)
        mask_array[100:120, 200:230] = 255
        mask = Image.fromarray(mask_array)

        with patch("wmcleaner.inpaint.lama_available", return_value=True):
            inpainter = Inpainter("lama", "cpu")
        inpainter._lama = _PaddedLama()

        result = inpainter.run(source, mask)
        result_array = np.asarray(result)

        self.assertEqual(result.size, source.size)
        self.assertTrue(np.all(result_array[0, 0] == np.array([20, 40, 60])))
        self.assertTrue(np.all(result_array[105, 205] == np.array([255, 0, 0])))

    def test_mask_size_is_normalized_to_image(self) -> None:
        source = Image.new("RGB", (850, 1205), (20, 40, 60))
        mismatched_mask = Image.new("L", (856, 1208), 0)
        mismatched_mask.putpixel((100, 100), 255)

        with patch("wmcleaner.inpaint.lama_available", return_value=True):
            inpainter = Inpainter("lama", "cpu")
        inpainter._lama = _PaddedLama()

        result = inpainter.run(source, mismatched_mask)
        self.assertEqual(result.size, source.size)


if __name__ == "__main__":
    unittest.main()
