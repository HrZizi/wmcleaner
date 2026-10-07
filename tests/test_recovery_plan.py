from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from wmcleaner.detector import Detection
from wmcleaner.recovery_plan import (
    _fit_surface_evidence,
    _robust_fit,
    build_overlay_plans,
    evaluate_registered_residual,
)


class RecoveryPlanTests(unittest.TestCase):
    def test_modern_constant_fit_resists_minority_exterior_artwork(self) -> None:
        box = (80, 50, 240, 100)
        for color in ((255, 255, 255), (235, 231, 225)):
            with self.subTest(color=color):
                page = np.full((230, 400, 3), color, dtype=np.uint8)
                # A smooth neighboring panel occupies a minority of the ring.
                # Its border is excluded by the gradient gate, its interior is not.
                page[158:168, 80:320] = (202, 200, 198)
                evidence, prediction = _fit_surface_evidence(page, box, "modern")
                self.assertEqual(evidence.model, "constant")
                np.testing.assert_array_equal(prediction[80, 150], color)
                self.assertLess(evidence.constant_seam_error, 0.01)

    def test_constant_surface_rejects_systematic_perimeter_bias(self) -> None:
        from unittest.mock import patch
        page = np.full((230, 400, 3), 249, dtype=np.uint8)
        box = (80, 50, 240, 100)
        # Wide-ring majority differs from the immediate exterior surface.
        page[48:152, 78:322] = 255
        # Isolate acceptance from the independent affine fallback.
        original = _robust_fit
        def fit(coords, values, model, **kwargs):
            result = original(coords, values, model, **kwargs)
            if model == "affine" and result is not None:
                coefficients, design = result
                coefficients = coefficients.copy(); coefficients[0] = 200
                return coefficients, design
            return result
        with patch("wmcleaner.recovery_plan._robust_fit", side_effect=fit):
            evidence, _ = _fit_surface_evidence(page, box, "modern")
        self.assertLess(evidence.constant_holdout_error, 14)
        self.assertLess(evidence.constant_seam_error, 16)
        self.assertFalse(evidence.valid)

    @staticmethod
    def _modern_detection(shape: tuple[int, int], box: tuple[int, int, int, int]) -> Detection:
        mask = np.zeros(shape, dtype=np.uint8)
        x, y, w, h = box
        mask[y:y + h, x:x + w] = 255
        return Detection("baozimh_logo", 0.96, box, mask)

    def test_constant_and_affine_surface_selection_use_boundary_holdout(self) -> None:
        h, w = 300, 520
        box = (100, 50, 300, 120)
        detection = self._modern_detection((h, w), box)

        constant = np.full((h, w, 3), 235, dtype=np.uint8)
        constant_plans, _ = build_overlay_plans(
            Image.fromarray(constant),
            [detection],
        )
        self.assertEqual(constant_plans[0].surface.model, "constant")
        self.assertTrue(constant_plans[0].surface.valid)

        yy, xx = np.indices((h, w))
        gradient = np.clip(
            218.0 + 22.0 * xx / w + 10.0 * yy / h,
            0,
            255,
        ).astype(np.uint8)
        gradient_rgb = np.repeat(gradient[:, :, None], 3, axis=2)
        affine_plans, _ = build_overlay_plans(
            Image.fromarray(gradient_rgb),
            [detection],
        )
        self.assertEqual(affine_plans[0].surface.model, "affine")
        self.assertGreater(affine_plans[0].surface.complexity_gain, 0.15)

    def test_boundary_connected_low_contrast_structure_is_trusted(self) -> None:
        h, w = 220, 500
        box = (100, 40, 300, 100)
        page = np.full((h, w, 3), 242, dtype=np.uint8)
        # A low-contrast source contour crosses both sides of the aligned box.
        page[88:91, 40:460] = 218
        detection = self._modern_detection((h, w), box)
        plans, _ = build_overlay_plans(Image.fromarray(page), [detection])
        plan = plans[0]

        self.assertGreater(int(np.count_nonzero(plan.trusted_source_mask)), 0)
        self.assertTrue(np.any(plan.trusted_source_mask[89, 100:400] > 0))
        self.assertTrue(
            np.all(plan.missing_information_mask[plan.trusted_source_mask > 0] == 0)
        )

    def test_plans_are_independent_when_a_second_detection_is_added(self) -> None:
        h, w = 260, 700
        first_box = (40, 20, 260, 90)
        second_box = (380, 150, 260, 90)
        page = np.full((h, w, 3), 235, dtype=np.uint8)
        page[150:240, 380:640] = (45, 55, 75)
        first = self._modern_detection((h, w), first_box)
        second = self._modern_detection((h, w), second_box)

        alone, _ = build_overlay_plans(Image.fromarray(page), [first])
        together, _ = build_overlay_plans(Image.fromarray(page), [first, second])

        self.assertEqual(alone[0].route, together[0].route)
        self.assertEqual(alone[0].scene.scene_class, together[0].scene.scene_class)
        self.assertEqual(
            int(np.count_nonzero(alone[0].missing_information_mask)),
            int(np.count_nonzero(together[0].missing_information_mask)),
        )

    def test_registered_residual_requires_all_three_gates(self) -> None:
        h, w = 140, 240
        box = (40, 40, 100, 40)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[50:70, 50:150] = 255  # area 2,000
        detection = Detection("baozimh_logo", 0.96, box, mask)

        dark = np.full((h, w, 3), 60, dtype=np.uint8)
        dark[55:65, 60:90] = 190  # 300 hits, 15% coverage
        strong = evaluate_registered_residual(Image.fromarray(dark), [detection])
        self.assertTrue(strong.should_flag)
        self.assertGreater(int(np.count_nonzero(strong.operation_mask)), 0)
        self.assertTrue(all(item["qualifies"] for item in strong.components))

        speck = np.full((h, w, 3), 60, dtype=np.uint8)
        speck[55:57, 60:65] = 190  # only 10 hits: below support gate
        weak = evaluate_registered_residual(Image.fromarray(speck), [detection])
        self.assertFalse(weak.should_flag)
        self.assertEqual(int(np.count_nonzero(weak.operation_mask)), 0)

        texture = np.full((h, w, 3), 60, dtype=np.uint8)
        texture[55:65, 60:90] = 75  # enough coverage, but weak local contrast
        weak_texture = evaluate_registered_residual(
            Image.fromarray(texture), [detection]
        )
        self.assertFalse(weak_texture.should_flag)

    def test_ambiguous_or_unsupported_geometry_never_routes_to_surface_fill(self) -> None:
        h, w = 260, 520
        page = np.full((h, w, 3), 242, dtype=np.uint8)
        # Two competing partial arcs are deliberately insufficient for a
        # deterministic circle continuation.  The planner may classify this as
        # structured or ambiguous, but must not promote a surface fill.
        import cv2

        cv2.ellipse(page, (190, 130), (130, 100), 0, 190, 245, (90, 90, 90), 2)
        cv2.ellipse(page, (335, 130), (130, 100), 0, 295, 350, (90, 90, 90), 2)
        box = (80, 60, 340, 120)
        detection = self._modern_detection((h, w), box)
        plans, _ = build_overlay_plans(Image.fromarray(page), [detection])
        plan = plans[0]
        if plan.scene.geometry_kind in {"circle", "ambiguous"}:
            self.assertNotIn(plan.route, {"surface-constant", "surface-affine"})
        self.assertNotEqual(plan.routing.generation_scope, "deterministic")


if __name__ == "__main__":
    unittest.main()
