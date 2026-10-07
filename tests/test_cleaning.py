from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from wmcleaner.cleaning import (
    CleanSettings,
    Cleaner,
    _compose_clipped_bubble_repair,
    _restore_circular_outline,
    _dense_logo_spans_dark_artwork,
    _legacy_banner_overlaps_clipped_bubble,
    _legacy_banner_cleanup_mask,
    _legacy_registration_supports_full_banner_cleanup,
    _modern_logo_has_uniform_pale_background,
    _fill_uniform_modern_logos,
    _modern_logo_scene_is_simple,
    _polish_modern_logo_text,
    _prepare_dense_overlap_hybrid,
)
from wmcleaner.detector import Detection, DetectionEvidence
from wmcleaner.legacy_banner import _boundary_seam_mask


class _FakeInpainter:
    masks: list[np.ndarray] = []

    def __init__(self, backend: str, device: str) -> None:
        self.backend = "lama" if backend in {"auto", "lama"} else backend

    def run(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        type(self).masks.append(np.asarray(mask, dtype=np.uint8).copy())
        return image.copy()


class CleaningTests(unittest.TestCase):
    def test_registration_conflict_holds_even_a_valid_surface_without_inference(self) -> None:
        page = Image.new("RGB", (450, 230), (242, 242, 242))
        mask = np.zeros((230, 450), dtype=np.uint8)
        mask[50:150, 80:320] = 255
        detection = Detection("baozimh_logo", .99, (80, 50, 240, 100), mask,
            DetectionEvidence(family="modern-logo", variant="current",
                registration_confidence=.78,
                reasons=("uncertain_registration", "variant_anchor_conflict")))
        cleaner = Cleaner.__new__(Cleaner)
        cleaner.settings = CleanSettings()
        from unittest.mock import Mock
        cleaner.inpainter = Mock(backend="anime-lama")
        cleaner.comic_inpainter = None
        with patch("wmcleaner.cleaning.detect_watermarks", return_value=[detection]):
            result = cleaner.clean(page)
        cleaner.inpainter.run.assert_not_called()
        self.assertFalse(result.cleaned)
        self.assertEqual(result.plans[0].route, "qa-hold")
        self.assertIn("uncertain_registration", result.qa_reasons)
        self.assertFalse(np.any(result.operation_mask))
        np.testing.assert_array_equal(result.image, page)

    def _detection(self) -> Detection:
        mask = np.zeros((110, 320), dtype=np.uint8)
        mask[20:88, 8:82] = 255
        mask[35:78, 92:300] = 255
        return Detection("baozimh_logo", 0.95, (0, 0, 320, 110), mask)

    def test_full_legacy_banner_cleanup_requires_strong_aligned_registration(self) -> None:
        plan = SimpleNamespace(
            family="legacy",
            identity=SimpleNamespace(
                variant="legacy-banner",
                confidence=0.96,
            ),
            registration=SimpleNamespace(
                confidence=0.97,
                source="detector-evidence",
                offset=(0, 0),
                evidence=("aligned-detection-box",),
            ),
        )

        self.assertTrue(_legacy_registration_supports_full_banner_cleanup(plan))
        plan.registration.confidence = 0.89
        self.assertFalse(_legacy_registration_supports_full_banner_cleanup(plan))
        plan.registration.confidence = 0.97
        plan.registration.offset = (1, 0)
        self.assertFalse(_legacy_registration_supports_full_banner_cleanup(plan))

    def _dense_page(self) -> Image.Image:
        page = np.full((110, 320, 3), 45, dtype=np.uint8)
        # Yellow icon square; the helper localizes the largest connected yellow
        # component rather than assuming a fixed absolute size.
        page[20:88, 8:82] = (250, 205, 40)
        # Pale watermark-text-shaped area on dark artwork.
        page[35:78, 92:300] = (180, 180, 180)
        return Image.fromarray(page)

    def test_dense_overlap_hybrid_keeps_ai_mask_compact(self) -> None:
        image = self._dense_page()
        det = self._detection()
        prepared, ai_mask, report_mask = _prepare_dense_overlap_hybrid(image, [det])

        self.assertEqual(prepared.size, image.size)
        self.assertGreater(int(np.count_nonzero(ai_mask)), 0)
        ys, xs = np.where(ai_mask > 0)
        self.assertLess(int(xs.max()), 100)  # icon only, never the full text span
        self.assertGreater(int(np.count_nonzero(report_mask)), int(np.count_nonzero(ai_mask)))
        # Text region is repaired/reported without becoming generative AI area.
        self.assertEqual(int(ai_mask[55, 160]), 0)
        self.assertEqual(int(report_mask[55, 160]), 255)

    def test_dense_logo_darkness_selects_hybrid_only_for_dark_artwork(self) -> None:
        det = self._detection()
        self.assertTrue(_dense_logo_spans_dark_artwork(self._dense_page(), [det]))

        light_page = np.full((110, 320, 3), 225, dtype=np.uint8)
        light_page[20:88, 8:82] = (250, 205, 40)
        light_page[35:78, 92:300] = (180, 180, 180)
        self.assertFalse(
            _dense_logo_spans_dark_artwork(Image.fromarray(light_page), [det])
        )

    def test_legacy_banner_mask_fills_panel_but_preserves_non_watermark_dark_line(self) -> None:
        page = np.full((130, 360, 3), 220, dtype=np.uint8)
        page[45:48, 255:320] = 20
        core = np.zeros((130, 360), dtype=np.uint8)
        core[30:60, 30:90] = 255
        core[35:65, 110:250] = 255
        det = Detection("baozimh_top_left", 0.95, (20, 20, 325, 92), core)

        mask = _legacy_banner_cleanup_mask(Image.fromarray(page), [det])

        self.assertEqual(int(mask[30, 300]), 255)  # translucent panel background
        self.assertEqual(int(mask[46, 280]), 0)    # real dark comic structure
        self.assertEqual(int(mask[45, 120]), 255)  # watermark core still removed

    def test_legacy_boundary_seam_wraps_all_four_edges_and_corners(self) -> None:
        core = np.zeros((150, 400), dtype=np.uint8)
        det = Detection("baozimh_top_left", 0.96, (30, 25, 325, 92), core)

        seam = _boundary_seam_mask((150, 400), det, half_width=20)

        self.assertEqual(int(seam[25, 160]), 255)   # top
        self.assertEqual(int(seam[114, 160]), 255)  # bottom
        self.assertEqual(int(seam[65, 30]), 255)    # left
        self.assertEqual(int(seam[65, 339]), 255)   # right
        self.assertEqual(int(seam[28, 34]), 255)    # rounded corner

    def test_legacy_banner_auto_falls_back_to_unpolished_solid_repair(self) -> None:
        image = Image.new("RGB", (360, 130), (230, 230, 230))
        core = np.zeros((130, 360), dtype=np.uint8)
        core[30:60, 30:90] = 255
        det = Detection("baozimh_top_left", 0.96, (20, 20, 325, 92), core)
        _FakeInpainter.masks.clear()

        with patch("wmcleaner.cleaning.Inpainter", _FakeInpainter), patch(
            "wmcleaner.cleaning.detect_watermarks", return_value=[det]
        ), patch(
            "wmcleaner.cleaning.max_structural_overlap_score", return_value=0.0
        ):
            cleaner = Cleaner(CleanSettings(backend="lama"))
            result = cleaner.clean(image)

        self.assertTrue(result.used_legacy_banner_mask)
        self.assertTrue(result.used_comic_fallback)
        self.assertEqual(result.backend, "anime-lama")
        self.assertEqual(result.legacy_banner_mode_used, "solid")
        self.assertEqual(
            result.legacy_fallback_reason,
            "panel-fit-did-not-improve-boundary-polish-skipped",
        )
        self.assertGreater(
            int(np.count_nonzero(_FakeInpainter.masks[-1])),
            int(np.count_nonzero(core)),
        )

    def test_legacy_banner_solid_mode_keeps_v0314_compatibility(self) -> None:
        image = Image.new("RGB", (360, 130), (230, 230, 230))
        core = np.zeros((130, 360), dtype=np.uint8)
        core[30:60, 30:90] = 255
        det = Detection("baozimh_top_left", 0.96, (20, 20, 325, 92), core)
        _FakeInpainter.masks.clear()

        with patch("wmcleaner.cleaning.Inpainter", _FakeInpainter), patch(
            "wmcleaner.cleaning.detect_watermarks", return_value=[det]
        ), patch(
            "wmcleaner.cleaning.max_structural_overlap_score", return_value=0.0
        ):
            cleaner = Cleaner(
                CleanSettings(backend="lama", legacy_banner_mode="solid")
            )
            result = cleaner.clean(image)

        self.assertEqual(result.legacy_banner_mode_used, "solid")
        self.assertGreater(
            int(np.count_nonzero(_FakeInpainter.masks[-1])),
            int(np.count_nonzero(core)),
        )

    def test_uniform_pale_legacy_banner_uses_fill_without_inpainting(self) -> None:
        page = np.full((170, 850, 3), 255, dtype=np.uint8)
        core = np.zeros((170, 850), dtype=np.uint8)
        core[35:80, 540:800] = 255
        det = Detection("baozimh_top_left", 0.98, (520, 20, 325, 92), core)
        _FakeInpainter.masks.clear()

        with patch("wmcleaner.cleaning.Inpainter", _FakeInpainter), patch(
            "wmcleaner.cleaning.detect_watermarks", return_value=[det]
        ):
            result = Cleaner(CleanSettings(backend="lama")).clean(
                Image.fromarray(page)
            )

        self.assertEqual(result.backend, "solid-fill")
        self.assertEqual(result.legacy_banner_mode_used, "uniform-fill")
        self.assertFalse(_FakeInpainter.masks)
        self.assertTrue(np.all(np.asarray(result.image)[20:112, 520:845] == 255))
        self.assertEqual(int(np.count_nonzero(result.mask)), 325 * 92)

    def test_uniform_pale_legacy_fill_preserves_dark_bubble_arc(self) -> None:
        page = np.full((170, 850, 3), 255, dtype=np.uint8)
        page[62:66, 520:845] = 18
        core = np.zeros((170, 850), dtype=np.uint8)
        core[35:80, 540:800] = 255
        det = Detection("baozimh_top_left", 0.98, (520, 20, 325, 92), core)

        with patch("wmcleaner.cleaning.Inpainter", _FakeInpainter), patch(
            "wmcleaner.cleaning.detect_watermarks", return_value=[det]
        ):
            result = Cleaner(CleanSettings(backend="lama")).clean(Image.fromarray(page))

        cleaned = np.asarray(result.image)
        self.assertTrue(np.all(cleaned[63, 600] == 18))
        self.assertEqual(int(result.mask[63, 600]), 0)
        self.assertTrue(np.all(cleaned[45, 700] == 255))

    def test_uniform_pale_modern_logo_preserves_text_and_colored_art(self) -> None:
        page = np.full((170, 850, 3), 255, dtype=np.uint8)
        page[42:112, 110:190] = (248, 205, 35)
        page[70:74, 215:470] = 20
        page[85:95, 220:500] = (220, 65, 85)
        core = np.zeros((170, 850), dtype=np.uint8)
        core[42:112, 110:190] = 255
        core[80:105, 190:430] = 255
        det = Detection("baozimh_logo", 0.98, (100, 30, 320, 110), core)
        image = Image.fromarray(page)

        self.assertTrue(_modern_logo_has_uniform_pale_background(image, det))
        cleaned, mask = _fill_uniform_modern_logos(image, [det])
        result = np.asarray(cleaned)

        self.assertTrue(np.all(result[72, 300] == 20))
        self.assertTrue(np.all(result[90, 300] == (220, 65, 85)))
        self.assertTrue(np.all(result[65, 145] == 255))
        self.assertEqual(int(mask[72, 300]), 0)
        self.assertEqual(int(mask[65, 145]), 255)

    def test_uniform_pale_modern_logo_preserves_dark_ink_inside_text_mask(self) -> None:
        page = np.full((150, 360, 3), 255, dtype=np.uint8)
        page[35:105, 35:110] = (248, 205, 35)
        page[76:78, 135:250] = 20
        core = np.zeros((150, 360), dtype=np.uint8)
        core[35:105, 35:110] = 255
        core[60:95, 125:300] = 255
        det = Detection("baozimh_logo", 0.98, (0, 20, 350, 110), core)

        cleaned, mask = _fill_uniform_modern_logos(Image.fromarray(page), [det])
        result = np.asarray(cleaned)

        self.assertTrue(np.all(result[76:78, 135:250] == 20))
        self.assertTrue(np.all(mask[76:78, 135:250] == 0))

    def test_uniform_pale_modern_logo_bridges_sparse_collinear_gaps(self) -> None:
        page = np.full((150, 360, 3), 255, dtype=np.uint8)
        page[35:105, 35:110] = (248, 205, 35)
        page[60, 125:145] = 20
        page[60, 154:174] = 20
        page[78, 130:140] = 20
        page[78, 145:149] = 20
        page[92, 138:158] = 20
        core = np.zeros((150, 360), dtype=np.uint8)
        core[35:105, 35:110] = 255
        core[58:63, 140:160] = 255
        det = Detection("baozimh_logo", 0.98, (0, 20, 350, 110), core)

        cleaned, mask = _fill_uniform_modern_logos(Image.fromarray(page), [det])
        result = np.asarray(cleaned)

        self.assertTrue(np.all(result[60, 145:154] < 105))
        self.assertTrue(np.all(mask[60, 145:154] == 0))
        self.assertTrue(np.all(result[78, 140:145] < 105))
        self.assertTrue(np.all(result[70, 125:174] == 255))

    def test_circle_repair_refines_hough_guess_to_surviving_arc(self) -> None:
        source = np.full((400, 400, 3), 255, dtype=np.uint8)
        cv2.circle(source, (180, 200), 100, (20, 20, 20), 3, lineType=cv2.LINE_AA)
        repair = np.zeros((400, 400), dtype=np.uint8)
        repair[170:231, 274:292] = 255
        result = source.copy()
        result[repair > 0] = 255
        detection = Detection("baozimh_logo", 0.98, (250, 150, 80, 80), repair)
        # Crop origin is (110, 6), so the true local circle is (70, 194, 100).
        rough = np.array([[[72.0, 194.0, 101.0]]], dtype=np.float32)

        with patch("wmcleaner.cleaning.cv2.HoughLinesP", return_value=None), patch(
            "wmcleaner.cleaning.cv2.HoughCircles",
            return_value=rough,
        ):
            _restore_circular_outline(source, result, repair, detection)

        self.assertLess(int(np.mean(result[198:203, 278:282])), 80)
        self.assertGreater(int(np.mean(result[198:203, 285:289])), 220)

    def test_clipped_circle_bridge_meets_both_observed_arcs_flush(self) -> None:
        source = np.full((130, 400, 3), 255, dtype=np.uint8)
        cv2.circle(source, (200, 200), 150, (20, 20, 20), 2, lineType=cv2.LINE_AA)
        repair = np.zeros((130, 400), dtype=np.uint8)
        repair[25:120, 100:301] = 255
        result = source.copy()
        result[repair > 0] = 255
        detection = Detection("baozimh_top_left", 0.98, (100, 25, 200, 95), repair)
        rough = np.array([[[204.0, 203.0, 151.0]]], dtype=np.float32)

        with patch("wmcleaner.cleaning.cv2.HoughLinesP", return_value=None), patch(
            "wmcleaner.cleaning.cv2.HoughCircles",
            return_value=rough,
        ):
            _restore_circular_outline(source, result, repair, detection)

        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        for column in range(96, 305):
            self.assertTrue(np.any(gray[:, column] < 105), column)

        for column in list(range(92, 112)) + list(range(289, 309)):
            rows = np.flatnonzero(gray[:, column] < 105)
            expected = 200.0 - np.sqrt(150.0**2 - (column - 200.0) ** 2)
            self.assertLessEqual(float(np.min(np.abs(rows - expected))), 2.5)

    def test_sparse_curve_on_pale_background_prefers_model_repair(self) -> None:
        page = np.full((150, 360, 3), 255, dtype=np.uint8)
        page[35:105, 35:110] = (248, 205, 35)
        cv2.ellipse(page, (260, 110), (100, 100), 0, 190, 230, (20, 20, 20), 2)
        core = np.zeros((150, 360), dtype=np.uint8)
        core[35:105, 35:110] = 255
        det = Detection("baozimh_logo", 0.98, (0, 20, 350, 110), core)

        self.assertFalse(
            _modern_logo_has_uniform_pale_background(Image.fromarray(page), det)
        )

    def test_uniform_pale_modern_logo_bridges_long_panel_border(self) -> None:
        page = np.full((150, 360, 3), 255, dtype=np.uint8)
        page[35:105, 35:110] = (248, 205, 35)
        page[88:91, 8:350] = 20
        core = np.zeros((150, 360), dtype=np.uint8)
        core[35:105, 35:110] = 255
        core[70:105, 100:220] = 255
        det = Detection("baozimh_logo", 0.98, (0, 20, 350, 110), core)

        cleaned, mask = _fill_uniform_modern_logos(Image.fromarray(page), [det])
        result = np.asarray(cleaned)

        self.assertTrue(np.all(result[89, 8:350] == 20))
        self.assertTrue(np.all(mask[89, 8:350] == 0))

    def test_uniform_pale_modern_logo_does_not_join_scattered_text(self) -> None:
        page = np.full((150, 360, 3), 255, dtype=np.uint8)
        page[35:105, 35:110] = (248, 205, 35)
        for start in (125, 175, 225, 275):
            page[70:74, start:start + 18] = 20
        core = np.zeros((150, 360), dtype=np.uint8)
        core[35:105, 35:110] = 255
        det = Detection("baozimh_logo", 0.98, (0, 20, 350, 110), core)

        cleaned, _ = _fill_uniform_modern_logos(Image.fromarray(page), [det])

        self.assertTrue(np.all(np.asarray(cleaned)[71, 150:170] == 255))

    def test_uniform_pale_modern_logo_requires_strong_circle_support(self) -> None:
        page = np.full((150, 360, 3), 255, dtype=np.uint8)
        page[35:105, 35:110] = (248, 205, 35)
        cv2.ellipse(page, (260, 110), (100, 100), 0, 190, 230, (20, 20, 20), 2)
        core = np.zeros((150, 360), dtype=np.uint8)
        core[35:105, 35:110] = 255
        det = Detection("baozimh_logo", 0.98, (0, 20, 350, 110), core)

        with patch("wmcleaner.cleaning._restore_circular_outline") as restore:
            _fill_uniform_modern_logos(Image.fromarray(page), [det])

        self.assertEqual(restore.call_count, 1)
        self.assertEqual(restore.call_args.kwargs["min_support"], 0.23)

    def test_logo_text_polish_expands_only_confirmed_template_component(self) -> None:
        source = np.full((110, 320, 3), (42, 62, 92), dtype=np.uint8)
        result = source.copy()
        result[30:58, 132:142] = (165, 168, 172)
        result[30:58, 210:250] = (205, 65, 85)
        template = np.zeros((110, 320), dtype=np.uint8)
        template[30:58, 120:160] = 255
        template[30:58, 210:250] = 255
        det = Detection("baozimh_logo", 0.98, (0, 0, 320, 110), template)

        with patch("wmcleaner.cleaning._logo_variant_mask", return_value=template):
            polished = _polish_modern_logo_text(
                Image.fromarray(result),
                Image.fromarray(source),
                [det],
            )

        cleaned = np.asarray(polished)
        self.assertLess(int(cleaned[44, 140, 0]), 100)
        self.assertTrue(np.all(cleaned[44, 230] == (205, 65, 85)))

    def test_nonuniform_top_right_legacy_banner_uses_coherent_mask(self) -> None:
        page = np.full((170, 850, 3), (180, 170, 195), dtype=np.uint8)
        page[:, 700:] = 250
        core = np.zeros((170, 850), dtype=np.uint8)
        core[35:80, 540:800] = 255
        det = Detection("baozimh_top_left", 0.98, (520, 20, 325, 92), core)
        _FakeInpainter.masks.clear()

        with patch("wmcleaner.cleaning.Inpainter", _FakeInpainter), patch(
            "wmcleaner.cleaning.detect_watermarks", return_value=[det]
        ), patch(
            "wmcleaner.cleaning.max_structural_overlap_score", return_value=0.0
        ):
            result = Cleaner(CleanSettings(backend="lama")).clean(
                Image.fromarray(page)
            )

        self.assertEqual(result.legacy_banner_mode_used, "coherent")
        self.assertEqual(len(_FakeInpainter.masks), 1)
        self.assertEqual(int(_FakeInpainter.masks[0][15, 515]), 255)
        self.assertEqual(int(_FakeInpainter.masks[0][116, 849]), 255)

    def test_dark_top_left_legacy_scene_selects_clipped_bubble_repair(self) -> None:
        page = np.full((170, 850, 3), (55, 65, 90), dtype=np.uint8)
        page[:40, 90:370] = 245
        core = np.zeros((170, 850), dtype=np.uint8)
        core[35:80, 40:300] = 255
        det = Detection("baozimh_top_left", 0.98, (20, 20, 325, 92), core)

        self.assertTrue(
            _legacy_banner_overlaps_clipped_bubble(Image.fromarray(page), [det])
        )

    def test_clipped_bubble_composition_keeps_bubble_and_clean_exterior(self) -> None:
        outside = np.full((170, 850, 3), (55, 65, 90), dtype=np.uint8)
        inside = outside.copy()
        polygon = np.array([[90, 0], [370, 0], [300, 62], [165, 62]])
        cv2.fillPoly(inside, [polygon], (245, 245, 245))
        core = np.zeros((170, 850), dtype=np.uint8)
        det = Detection("baozimh_top_left", 0.98, (20, 20, 325, 92), core)

        result = np.asarray(
            _compose_clipped_bubble_repair(
                Image.fromarray(inside),
                Image.fromarray(outside),
                det,
            )
        )

        self.assertTrue(np.all(result[25, 200] > 230))
        self.assertTrue(np.all(result[100, 200] == (55, 65, 90)))
        self.assertTrue(np.all(result[62, 220] < 40))

    def test_promo_only_page_uses_deterministic_white_fill_without_inpainting(self) -> None:
        page = np.full((240, 850, 3), 255, dtype=np.uint8)
        page[20:95, 315:390] = (248, 205, 35)
        page[35:70, 410:650] = 20
        mask = np.zeros((240, 850), dtype=np.uint8)
        mask[:201] = 255
        det = Detection("baozimh_promo", 0.98, (0, 0, 850, 201), mask)
        _FakeInpainter.masks.clear()

        with patch("wmcleaner.cleaning.Inpainter", _FakeInpainter), patch(
            "wmcleaner.cleaning.detect_watermarks", return_value=[det]
        ):
            result = Cleaner(CleanSettings(backend="lama")).clean(
                Image.fromarray(page)
            )

        self.assertTrue(result.cleaned)
        self.assertEqual(result.backend, "solid-fill")
        self.assertFalse(_FakeInpainter.masks)
        self.assertTrue(np.all(np.asarray(result.image)[:201] == 255))
        self.assertEqual(int(np.count_nonzero(result.mask)), 850 * 201)

    def test_simple_logo_scene_prefers_coherent_repair(self) -> None:
        page = np.full((110, 320, 3), 210, dtype=np.uint8)
        self.assertTrue(
            _modern_logo_scene_is_simple(Image.fromarray(page), [self._detection()])
        )

    def test_sparse_panel_lines_do_not_use_destructive_coherent_repair(self) -> None:
        page = np.full((110, 320, 3), 240, dtype=np.uint8)
        page[20:88, 8:82] = (250, 205, 40)
        page[:, 150:154] = 25
        page[82:86, 100:] = 25
        self.assertFalse(
            _modern_logo_scene_is_simple(Image.fromarray(page), [self._detection()])
        )

    def test_detailed_logo_scene_keeps_dark_structure_protection(self) -> None:
        page = np.full((110, 320, 3), 230, dtype=np.uint8)
        for row in range(3):
            for col in range(5):
                y = 8 + row * 30
                x = 10 + col * 58
                page[y:y + 8, x:x + 14] = 20
        self.assertFalse(
            _modern_logo_scene_is_simple(Image.fromarray(page), [self._detection()])
        )

    def test_high_overlap_uses_dense_hybrid_not_full_solid_ai_mask(self) -> None:
        image = self._dense_page()
        det = self._detection()
        captured: dict[str, bool] = {}
        _FakeInpainter.masks.clear()

        def fake_combine(*args, **kwargs):
            captured["protect_dark"] = kwargs["protect_dark"]
            return np.full((110, 320), 255, dtype=np.uint8)

        with patch("wmcleaner.cleaning.Inpainter", _FakeInpainter), patch(
            "wmcleaner.cleaning.detect_watermarks", return_value=[det]
        ), patch(
            "wmcleaner.cleaning.max_structural_overlap_score", return_value=0.35
        ), patch(
            "wmcleaner.cleaning.combine_cleanup_masks", side_effect=fake_combine
        ):
            cleaner = Cleaner(CleanSettings(backend="lama", comic_dense_overlap_threshold=0.18))
            result = cleaner.clean(image)

        self.assertTrue(captured["protect_dark"])
        self.assertTrue(result.used_dense_logo_mask)
        self.assertEqual(result.dense_repair_mode, "hybrid")
        self.assertTrue(result.used_comic_fallback)
        self.assertEqual(result.backend, "anime-lama")
        self.assertTrue(_FakeInpainter.masks)
        # The Anime-LaMa mask is the small icon box, not a 320x110 rectangle.
        self.assertLess(int(np.count_nonzero(_FakeInpainter.masks[-1])), 12000)
        self.assertEqual(int(_FakeInpainter.masks[-1][55, 160]), 0)

    def test_light_high_overlap_uses_precise_anime_lama_mask(self) -> None:
        page = np.full((110, 320, 3), 225, dtype=np.uint8)
        page[20:88, 8:82] = (250, 205, 40)
        page[35:78, 92:300] = (180, 180, 180)
        image = Image.fromarray(page)
        det = self._detection()
        captured: dict[str, str] = {}
        _FakeInpainter.masks.clear()

        def fake_combine(*args, **kwargs):
            captured["logo_style"] = kwargs["logo_style"]
            return det.mask.copy()

        with patch("wmcleaner.cleaning.Inpainter", _FakeInpainter), patch(
            "wmcleaner.cleaning.detect_watermarks", return_value=[det]
        ), patch(
            "wmcleaner.cleaning.max_structural_overlap_score", return_value=0.24
        ), patch(
            "wmcleaner.cleaning.combine_cleanup_masks", side_effect=fake_combine
        ):
            cleaner = Cleaner(
                CleanSettings(backend="lama", comic_dense_overlap_threshold=0.18)
            )
            result = cleaner.clean(image)

        self.assertTrue(result.used_dense_logo_mask)
        self.assertEqual(result.dense_repair_mode, "precise")
        self.assertEqual(captured["logo_style"], "precise")
        self.assertTrue(np.array_equal(_FakeInpainter.masks[-1], det.mask))

    def test_moderate_overlap_keeps_normal_protected_mask(self) -> None:
        page = np.full((110, 320, 3), 225, dtype=np.uint8)
        page[20:88, 8:82] = (250, 205, 40)
        page[35:78, 92:300] = (180, 180, 180)
        image = Image.fromarray(page)
        det = self._detection()
        captured: dict[str, bool] = {}

        def fake_combine(*args, **kwargs):
            if "protect_dark" in kwargs:
                captured["protect_dark"] = kwargs["protect_dark"]
            return np.full((110, 320), 255, dtype=np.uint8)

        with patch("wmcleaner.cleaning.Inpainter", _FakeInpainter), patch(
            "wmcleaner.cleaning.detect_watermarks", return_value=[det]
        ), patch(
            "wmcleaner.cleaning.max_structural_overlap_score", return_value=0.11
        ), patch(
            "wmcleaner.cleaning._modern_logo_scene_is_simple", return_value=False
        ), patch(
            "wmcleaner.cleaning.combine_cleanup_masks", side_effect=fake_combine
        ):
            cleaner = Cleaner(CleanSettings(backend="lama", comic_dense_overlap_threshold=0.18))
            result = cleaner.clean(image)

        self.assertTrue(captured["protect_dark"])
        self.assertFalse(result.used_dense_logo_mask)
        self.assertTrue(result.used_comic_fallback)

    def test_dark_logo_artwork_does_not_force_hybrid_below_overlap_threshold(self) -> None:
        image = self._dense_page()
        det = self._detection()
        _FakeInpainter.masks.clear()

        with patch("wmcleaner.cleaning.Inpainter", _FakeInpainter), patch(
            "wmcleaner.cleaning.detect_watermarks", return_value=[det]
        ), patch(
            "wmcleaner.cleaning.max_structural_overlap_score", return_value=0.03
        ):
            result = Cleaner(
                CleanSettings(backend="lama", comic_dense_overlap_threshold=0.18)
            ).clean(image)

        self.assertFalse(result.used_dense_logo_mask)
        self.assertIsNone(result.dense_repair_mode)
        self.assertEqual(result.backend, "anime-lama")


if __name__ == "__main__":
    unittest.main()
