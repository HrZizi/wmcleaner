from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from wmcleaner.detector import (
    Detection,
    DetectionEvidence,
    _extend_logo_mask_palette_halo,
    _load_gray_asset,
    _logo_dark_glyph_valid,
    _logo_variant_joint_support,
    _logo_registration_anchor_retention,
    _select_logo_variant,
    combine_cleanup_masks,
    detect_watermarks,
)


class DetectorTests(unittest.TestCase):
    def test_registration_checks_instance_anchor_with_phase_tolerance(self) -> None:
        anchor = _load_gray_asset("baozimh_bottom_right_anchor.png")
        yy, xx = np.nonzero(anchor)
        anchor = anchor[yy.min():yy.max()+1, xx.min():xx.max()+1].astype(np.float32) / 255
        for origin in ((30, 25), (117, 81)):
            yellow = np.zeros((230, 320), dtype=np.float32)
            x, y = origin
            yellow[y:y+anchor.shape[0], x:x+anchor.shape[1]] = anchor
            self.assertAlmostEqual(_logo_registration_anchor_retention(yellow, anchor, origin, (1, -2), 2), 1)
            self.assertLess(_logo_registration_anchor_retention(yellow, anchor, origin, (7, 7), 2), .85)

    @staticmethod
    def _page_with_logo_anchor(*, x: int, edge: str, width: int = 850, height: int = 1400) -> Image.Image:
        page = np.full((height, width, 3), 255, dtype=np.uint8)
        full_anchor = _load_gray_asset("baozimh_bottom_right_anchor.png")
        ys, xs = np.where(full_anchor > 0)
        ax0, ax1 = int(xs.min()), int(xs.max() + 1)
        ay0, ay1 = int(ys.min()), int(ys.max() + 1)

        # Use the bundled legacy-logo screenshot only for its icon pixels; the
        # modern mark uses the same dumpling icon but can move horizontally.
        from importlib import resources
        with resources.as_file(resources.files("wmcleaner").joinpath("templates", "baozimh_bottom_right.png")) as asset:
            source = np.array(Image.open(asset).convert("RGB"), dtype=np.uint8)
        icon = source[ay0:ay1, ax0:ax1]

        # The live site keeps the visible yellow icon about 34 px from the
        # corresponding page edge at the 850 px reference width.
        y = 34 if edge == "top" else height - 34 - icon.shape[0]
        page[y:y + icon.shape[0], x:x + icon.shape[1]] = icon
        return Image.fromarray(page, "RGB")

    def test_logo_glyph_validator_rejects_dark_outer_contour_without_inner_face(self) -> None:
        icon = np.full((75, 73), 220, dtype=np.uint8)
        # Similar overall dark share to the real icon, but all darkness sits on
        # an outer artwork contour instead of the dumpling face in the middle.
        icon[:, :6] = 40
        icon[:2, :] = 40
        self.assertFalse(_logo_dark_glyph_valid(icon))

    def test_logo_glyph_validator_accepts_real_bundled_icon(self) -> None:
        from importlib import resources
        full_anchor = _load_gray_asset("baozimh_bottom_right_anchor.png")
        ys, xs = np.where(full_anchor > 0)
        with resources.as_file(resources.files("wmcleaner").joinpath("templates", "baozimh_bottom_right.png")) as asset:
            source = np.array(Image.open(asset).convert("RGB"), dtype=np.uint8)
        icon = source[int(ys.min()):int(ys.max() + 1), int(xs.min()):int(xs.max() + 1)]
        gray = cv2.cvtColor(icon, cv2.COLOR_RGB2GRAY)
        self.assertTrue(_logo_dark_glyph_valid(gray))

    def test_modern_logo_can_move_horizontally_on_top_edge(self) -> None:
        image = self._page_with_logo_anchor(x=520, edge="top")
        detections = detect_watermarks(image)
        self.assertEqual([d.name for d in detections], ["baozimh_logo"])
        self.assertGreaterEqual(detections[0].confidence, 0.90)
        self.assertGreater(detections[0].box[0], 450)
        self.assertIsNotNone(detections[0].evidence)
        self.assertEqual(detections[0].evidence.glyph_mode, "strict")

    def test_detected_modern_logo_masks_complete_icon_including_dark_glyph(self) -> None:
        image = self._page_with_logo_anchor(x=520, edge="top")
        detection = detect_watermarks(image)[0]
        full_anchor = _load_gray_asset("baozimh_bottom_right_anchor.png")
        ys, xs = np.where(full_anchor > 0)
        x, y, _, _ = detection.box
        icon = detection.mask[
            y + int(ys.min()):y + int(ys.max() + 1),
            x + int(xs.min()):x + int(xs.max() + 1),
        ]
        self.assertTrue(np.all(icon == 255))

    def test_modern_logo_can_be_bottom_left(self) -> None:
        image = self._page_with_logo_anchor(x=33, edge="bottom")
        detections = detect_watermarks(image)
        self.assertEqual([d.name for d in detections], ["baozimh_logo"])
        self.assertLess(detections[0].box[0], 20)
        self.assertGreater(detections[0].box[1], 1200)

    def test_same_anchor_in_middle_of_page_is_not_a_watermark(self) -> None:
        page = np.full((1400, 850, 3), 255, dtype=np.uint8)
        full_anchor = _load_gray_asset("baozimh_bottom_right_anchor.png")
        ys, xs = np.where(full_anchor > 0)
        anchor = full_anchor[int(ys.min()):int(ys.max() + 1), int(xs.min()):int(xs.max() + 1)]
        page[650:650 + anchor.shape[0], 80:80 + anchor.shape[1]][anchor > 0] = (247, 204, 34)
        self.assertEqual(detect_watermarks(Image.fromarray(page, "RGB")), [])

    def test_plain_yellow_edge_rectangle_does_not_match_logo_shape(self) -> None:
        page = np.full((1400, 850, 3), 255, dtype=np.uint8)
        cv2.rectangle(page, (35, 34), (105, 104), (247, 204, 34), thickness=-1)
        self.assertEqual(detect_watermarks(Image.fromarray(page, "RGB")), [])

    def test_legacy_banner_can_move_to_top_right(self) -> None:
        from importlib import resources
        page = np.full((1000, 850, 3), 255, dtype=np.uint8)
        with resources.as_file(
            resources.files("wmcleaner").joinpath("templates", "baozimh_top_left.png")
        ) as asset:
            banner = np.asarray(Image.open(asset).convert("RGB"), dtype=np.uint8)
        page[20:20 + banner.shape[0], 520:520 + banner.shape[1]] = banner

        detections = detect_watermarks(Image.fromarray(page, "RGB"))

        legacy = [d for d in detections if d.name == "baozimh_top_left"]
        self.assertEqual(len(legacy), 1)
        self.assertGreaterEqual(legacy[0].confidence, 0.90)
        self.assertGreaterEqual(legacy[0].box[0], 500)

    def test_legacy_banner_is_searched_at_bottom_edge(self) -> None:
        from importlib import resources

        page = np.full((1000, 850, 3), 255, dtype=np.uint8)
        with resources.as_file(
            resources.files("wmcleaner").joinpath("templates", "baozimh_top_left.png")
        ) as asset:
            banner = np.asarray(Image.open(asset).convert("RGB"), dtype=np.uint8)
        y = page.shape[0] - 20 - banner.shape[0]
        page[y:y + banner.shape[0], 40:40 + banner.shape[1]] = banner

        detections = detect_watermarks(Image.fromarray(page, "RGB"))

        legacy = [d for d in detections if d.name == "baozimh_top_left"]
        self.assertEqual(len(legacy), 1)
        self.assertEqual(legacy[0].edge, "bottom")
        self.assertGreaterEqual(legacy[0].evidence.peak_margin, 0.01)
        self.assertGreaterEqual(legacy[0].box[1], y - 2)

    def test_detection_evidence_is_optional_and_mapping_compatible(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        legacy = Detection("test", 0.5, (0, 0, 8, 8), mask)
        self.assertIsNone(legacy.evidence)

        evidence = DetectionEvidence(
            family="modern-logo",
            variant="ambiguous",
            edge="top",
            anchor_score=0.94,
        )
        detected = Detection("baozimh_logo", 0.94, (0, 0, 8, 8), mask, evidence)
        self.assertEqual(detected.edge, "top")
        self.assertEqual(detected.variant, "ambiguous")
        self.assertEqual(evidence["identity"], "ambiguous")
        self.assertEqual(evidence["edge_evidence"], "top")
        self.assertEqual(evidence["anchor_score"], 0.94)

    def test_current_variant_joint_support_requires_wordmark_and_icon(self) -> None:
        roi = np.full((110, 320, 3), 120, dtype=np.uint8)
        current = _load_gray_asset("baozimh_logo_mask.png")
        anchor = _load_gray_asset("baozimh_bottom_right_anchor.png")

        # A visible pale current wordmark plus the matched icon supplies joint
        # support. The score is bounded by the independently measured icon
        # recall rather than allowing wordmark occupancy to stand in for it.
        roi[35:78, 110:315] = 220
        joint_support = _logo_variant_joint_support(
            roi,
            current,
            anchor,
            yellow_recall=0.62,
        )
        self.assertGreater(joint_support, 0.50)
        self.assertLessEqual(joint_support, 0.62)

        # No registered wordmark support cannot authorize the fallback even if
        # the icon score is otherwise high.
        roi[35:78, 110:315] = 120
        self.assertEqual(
            _logo_variant_joint_support(
                roi,
                current,
                anchor,
                yellow_recall=0.95,
            ),
            0.0,
        )

    def test_current_variant_occlusion_fallback_requires_joint_identity(self) -> None:
        image = self._page_with_logo_anchor(x=33, edge="bottom")
        current = _load_gray_asset("baozimh_logo_mask.png")

        current_selection = (
            current,
            "current",
            "current",
            0.581,
            0.45,
            0.08,
            0.073,
            (0, 0),
        )
        with patch("wmcleaner.detector._logo_dark_glyph_valid", return_value=False), patch(
            "wmcleaner.detector._yellow_shape_metrics",
            return_value=(0.85, 0.90),
        ), patch(
            "wmcleaner.detector._logo_dark_pattern_correlation",
            return_value=0.31,
        ), patch(
            "wmcleaner.detector._select_logo_variant",
            return_value=current_selection,
        ), patch(
            "wmcleaner.detector._logo_variant_joint_support",
            return_value=0.581,
        ):
            detections = detect_watermarks(image)

        modern = [d for d in detections if d.name == "baozimh_logo"]
        self.assertEqual(len(modern), 1)
        assert modern[0].evidence is not None
        self.assertEqual(modern[0].evidence.variant, "current")
        self.assertEqual(modern[0].evidence.mask_variant, "current")
        self.assertEqual(modern[0].evidence.glyph_mode, "occlusion-consistent")
        self.assertEqual(modern[0].evidence.joint_support, 0.581)
        self.assertIn("current_variant_joint_evidence", modern[0].evidence.reasons)

        # The same relaxed icon measurements remain rejected when variant
        # identity is ambiguous, which preserves the speech-text guard.
        ambiguous_selection = (
            current,
            "ambiguous",
            "common",
            0.581,
            0.55,
            0.04,
            0.030,
            (0, 0),
        )
        with patch("wmcleaner.detector._logo_dark_glyph_valid", return_value=False), patch(
            "wmcleaner.detector._yellow_shape_metrics",
            return_value=(0.85, 0.90),
        ), patch(
            "wmcleaner.detector._logo_dark_pattern_correlation",
            return_value=0.31,
        ), patch(
            "wmcleaner.detector._select_logo_variant",
            return_value=ambiguous_selection,
        ):
            self.assertEqual(detect_watermarks(image), [])

    def test_variant_identity_requires_margin_and_winner_gap(self) -> None:
        page = np.full((110, 320, 3), 245, dtype=np.uint8)
        current = _load_gray_asset("baozimh_logo_mask.png")
        older = _load_gray_asset("baozimh_bottom_right_mask.png")

        # A small support-vs-control margin, or a close competing margin, must
        # leave identity ambiguous and authorize common plus observed support.
        with patch(
            "wmcleaner.detector._variant_support_evidence",
            side_effect=[
                (0.80, 0.72, 0.08, (0, 0), current),
                (0.78, 0.72, 0.06, (0, 0), older),
            ],
        ):
            selected, identity, mask_variant, *_ = _select_logo_variant(
                page, (0, 0, 320, 110), search_radius=2
            )

        self.assertEqual(identity, "ambiguous")
        self.assertEqual(mask_variant, "common")
        self.assertTrue(np.array_equal(selected, cv2.bitwise_or(current, older)))

        # A registered winner is authorized only when both separation gates
        # pass; the returned offset is the winning local registration.
        with patch(
            "wmcleaner.detector._variant_support_evidence",
            side_effect=[
                (0.90, 0.20, 0.70, (1, -1), current),
                (0.40, 0.30, 0.10, (0, 0), older),
            ],
        ):
            selected, identity, mask_variant, support, competing, margin, gap, offset = (
                _select_logo_variant(page, (0, 0, 320, 110), search_radius=2)
            )

        self.assertEqual(identity, "current")
        self.assertEqual(mask_variant, "current")
        self.assertTrue(np.array_equal(selected, current))
        self.assertEqual((support, competing, margin, gap, offset), (0.90, 0.40, 0.70, 0.60, (1, -1)))

    def test_pale_speech_bubble_without_paired_hands_is_not_legacy_banner(self) -> None:
        page = np.full((1000, 850, 3), 250, dtype=np.uint8)
        cv2.ellipse(page, (180, 55), (155, 75), 0, 0, 360, (20, 20, 20), 4)
        for x in range(130, 231, 25):
            cv2.circle(page, (x, 55), 4, (20, 20, 20), thickness=-1)
        self.assertEqual(detect_watermarks(Image.fromarray(page, "RGB")), [])

    def test_opaque_centered_promo_marks_complete_white_boundary_strip(self) -> None:
        from importlib import resources
        page = np.full((1400, 850, 3), 255, dtype=np.uint8)
        full_anchor = _load_gray_asset("baozimh_bottom_right_anchor.png")
        ys, xs = np.where(full_anchor > 0)
        with resources.as_file(
            resources.files("wmcleaner").joinpath("templates", "baozimh_bottom_right.png")
        ) as asset:
            source = np.asarray(Image.open(asset).convert("RGB"), dtype=np.uint8)
        icon = source[int(ys.min()):int(ys.max() + 1), int(xs.min()):int(xs.max() + 1)]
        page[21:21 + icon.shape[0], 317:317 + icon.shape[1]] = icon
        cv2.rectangle(page, (405, 37), (610, 58), (20, 20, 20), thickness=-1)
        cv2.rectangle(page, (170, 120), (680, 133), (20, 20, 20), thickness=-1)
        cv2.rectangle(page, (250, 157), (600, 174), (20, 20, 20), thickness=-1)

        detections = detect_watermarks(Image.fromarray(page, "RGB"))

        promo = [d for d in detections if d.name == "baozimh_promo"]
        self.assertEqual(len(promo), 1)
        self.assertEqual(promo[0].box[0], 0)
        self.assertEqual(promo[0].box[2], 850)
        self.assertEqual(int(np.count_nonzero(promo[0].mask[:201])), 850 * 201)


    def test_logo_palette_halo_adds_faint_overlay_but_protects_black_border(self) -> None:
        page = np.full((110, 320, 3), 255, dtype=np.uint8)
        precise = np.zeros((110, 320), dtype=np.uint8)
        precise[40:60, 40:60] = 255

        # Faint low-saturation gray immediately outside the core should be
        # absorbed into the cleanup halo. A black panel border in the same halo
        # should remain protected.
        page[39, 45] = (235, 235, 235)
        page[40:60, 62] = (0, 0, 0)

        expanded = _extend_logo_mask_palette_halo(
            page, (0, 0, 320, 110), precise, halo_px=3
        )
        self.assertEqual(int(expanded[39, 45]), 255)
        self.assertEqual(int(expanded[50, 62]), 0)

    def test_modern_logo_mask_contains_anti_alias_halo(self) -> None:
        mask = _load_gray_asset("baozimh_logo_mask.png")
        ys, xs = np.where(mask > 0)
        self.assertGreaterEqual(int(np.count_nonzero(mask)), 11500)
        self.assertLessEqual(int(ys.min()), 12)
        self.assertGreaterEqual(int(ys.max()), 105)

    def test_occlusion_consistent_modern_fallback_requires_registered_shape_evidence(self) -> None:
        image = self._page_with_logo_anchor(x=520, edge="top")

        # Exercise the handoff independently from the strict glyph predicate:
        # the bundled icon still supplies the expected yellow shape and dark
        # pattern, while the strict validator is treated as occluded.
        with patch("wmcleaner.detector._logo_dark_glyph_valid", return_value=False):
            detections = detect_watermarks(image)

        modern = [d for d in detections if d.name == "baozimh_logo"]
        self.assertEqual(len(modern), 1)
        assert modern[0].evidence is not None
        self.assertEqual(modern[0].evidence.glyph_mode, "occlusion-consistent")
        self.assertGreaterEqual(modern[0].evidence.anchor_score, 0.93)
        self.assertGreaterEqual(modern[0].evidence.peak_margin, 0.01)
        self.assertGreaterEqual(modern[0].evidence.yellow_iou, 0.88)
        self.assertGreaterEqual(modern[0].evidence.yellow_recall, 0.90)
        self.assertGreaterEqual(modern[0].evidence.dark_pattern_correlation, 0.45)

    def test_solid_cleanup_mask_fills_logo_gaps_but_preserves_dark_structure(self) -> None:
        page = np.full((120, 340, 3), 240, dtype=np.uint8)
        # A dark panel border crossing the otherwise pale logo area.
        page[:, 105:108] = 0

        core = np.zeros((120, 340), dtype=np.uint8)
        core[30:80, 20:90] = 255
        core[40:70, 120:250] = 255
        det = Detection("baozimh_logo", 0.95, (0, 0, 300, 110), core)

        solid = combine_cleanup_masks(Image.fromarray(page), [det], logo_style="solid", padding=2)
        precise = combine_cleanup_masks(Image.fromarray(page), [det], logo_style="precise")

        # The gap between icon/text becomes part of one inpaint region in solid
        # mode, while precise mode keeps the gap untouched.
        self.assertEqual(int(precise[55, 105]), 0)
        self.assertEqual(int(solid[55, 95]), 255)
        # Genuine dark structure is punched back out.
        self.assertEqual(int(solid[55, 106]), 0)
        # Dark pixels that belong to the watermark core are still removed.
        page[50, 50] = 0
        solid_with_dark_core = combine_cleanup_masks(Image.fromarray(page), [det], logo_style="solid")
        self.assertEqual(int(solid_with_dark_core[50, 50]), 255)


    def test_solid_cleanup_mask_can_disable_dark_protection_for_dense_overlap(self) -> None:
        page = np.full((120, 340, 3), 40, dtype=np.uint8)
        core = np.zeros((120, 340), dtype=np.uint8)
        core[30:80, 20:90] = 255
        core[40:70, 120:250] = 255
        det = Detection("baozimh_logo", 0.95, (0, 0, 300, 110), core)

        protected = combine_cleanup_masks(
            Image.fromarray(page), [det], logo_style="solid", padding=2, protect_dark=True
        )
        dense = combine_cleanup_masks(
            Image.fromarray(page), [det], logo_style="solid", padding=2, protect_dark=False
        )

        # Normal mode preserves the dark background between watermark pieces.
        self.assertEqual(int(protected[55, 105]), 0)
        # Dense-overlap mode intentionally keeps one contiguous inpaint region.
        self.assertEqual(int(dense[55, 105]), 255)
        ys, xs = np.where(dense > 0)
        self.assertEqual(int(np.count_nonzero(dense)), (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1))

    def test_solid_cleanup_does_not_preserve_dark_ink_inside_watermark_core(self) -> None:
        page = np.full((120, 340, 3), 245, dtype=np.uint8)
        page[20:88, 8:82] = (250, 205, 40)
        page[52:55, 120:275] = 20
        core = np.zeros((120, 340), dtype=np.uint8)
        core[20:88, 8:82] = 255
        core[40:70, 110:290] = 255
        det = Detection("baozimh_logo", 0.95, (0, 0, 320, 110), core)

        solid = combine_cleanup_masks(Image.fromarray(page), [det], logo_style="solid")

        self.assertEqual(int(solid[53, 180]), 255)
        self.assertEqual(int(solid[53, 45]), 255)

    def test_solid_cleanup_preserves_non_yellow_colored_artwork(self) -> None:
        page = np.full((120, 340, 3), 245, dtype=np.uint8)
        page[20:88, 8:82] = (250, 205, 40)
        page[45:60, 95:280] = (225, 70, 90)
        core = np.zeros((120, 340), dtype=np.uint8)
        core[20:88, 8:82] = 255
        core[40:70, 90:295] = 255
        det = Detection("baozimh_logo", 0.95, (0, 0, 320, 110), core)

        solid = combine_cleanup_masks(Image.fromarray(page), [det], logo_style="solid")

        self.assertEqual(int(solid[52, 180]), 0)
        self.assertEqual(int(solid[50, 45]), 255)




if __name__ == "__main__":
    unittest.main()
