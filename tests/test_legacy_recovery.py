from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from wmcleaner.detector import Detection
from wmcleaner.legacy_banner import (
    LegacyBannerAssets,
    LegacyBannerFit,
    LegacyBannerRepair,
    LegacyRecoveryStability,
    _inverse_white_panel,
    _measure_recovery_stability,
    _recovery_is_eligible,
    compose_legacy_banner_repair,
    prepare_legacy_banner_repair,
)


class LegacyRecoveryContractTests(unittest.TestCase):
    @staticmethod
    def _synthetic_promoted_repair() -> tuple[np.ndarray, LegacyBannerRepair]:
        source = np.full((7, 9, 3), 180, dtype=np.uint8)
        source[3, 4] = (35, 35, 35)
        detection_mask = np.zeros((7, 9), dtype=np.uint8)
        detection_mask[3, 4] = 255
        detection = Detection(
            "baozimh_top_left",
            1.0,
            (0, 0, 9, 7),
            detection_mask,
        )

        panel = np.ones((7, 9), dtype=np.float32)
        foreground = np.zeros((7, 9), dtype=bool)
        foreground[3, 4] = True
        assets = LegacyBannerAssets(
            panel_shape=panel,
            foreground_support=foreground,
            foreground_core=np.zeros_like(foreground),
            foreground_suffix=np.zeros_like(foreground),
            foreground_offset=(0, 0),
        )
        candidate = np.full_like(source, 180)
        # A raw inverse which still contains the registered title foreground.
        candidate[3, 4] = (20, 20, 20)
        valid = np.ones((7, 9), dtype=bool)
        fit = LegacyBannerFit(
            working_space="encoded-rgb",
            alpha=0.70,
            baseline_error=10.0,
            fitted_error=2.0,
            improvement=0.80,
        )
        deblended = np.ones((7, 9), dtype=bool)
        deblended[3, 4] = False
        deblended[1, 1] = True
        deterministic = np.zeros((7, 9), dtype=bool)
        deterministic[3, 4] = True
        generated = np.zeros((7, 9), dtype=bool)
        stability = LegacyRecoveryStability(
            candidate_pixel_count=int(np.count_nonzero(deblended)),
            in_gamut_pixel_count=int(np.count_nonzero(deblended)),
            stable_pixel_count=int(np.count_nonzero(deblended)) - 1,
            in_gamut_fraction=1.0,
            stable_fraction=0.98,
            variation_median=1.0,
            variation_p90=2.0,
        )
        stable = deblended.astype(np.uint8) * 255
        # One otherwise recoverable inverse pixel is rejected by stability.
        stable[1, 1] = 0

        with (
            patch("wmcleaner.legacy_banner._legacy_banner_assets", return_value=assets),
            patch(
                "wmcleaner.legacy_banner._fit_panel_model",
                return_value=(candidate, valid, fit),
            ),
            patch(
                "wmcleaner.legacy_banner._legacy_local_repair_masks",
                return_value=(deblended, deterministic, generated),
            ),
            patch(
                "wmcleaner.legacy_banner._measure_recovery_stability",
                return_value=(stable, stability),
            ),
            patch(
                "wmcleaner.legacy_banner._LEGACY_SEMANTIC_RECOVERY_VALIDATED",
                True,
            ),
            patch("wmcleaner.legacy_banner._structural_pixel_mask", return_value=np.zeros((7, 9), dtype=np.uint8)),
        ):
            repair = prepare_legacy_banner_repair(
                Image.fromarray(source), [detection], mode="deblend"
            )
        return source, repair

    def test_promotion_requires_the_fit_improvement_gate(self) -> None:
        fit = LegacyBannerFit(
            working_space="encoded-rgb",
            alpha=0.72,
            baseline_error=10.0,
            fitted_error=8.0,
            improvement=0.19,
        )
        stability = LegacyRecoveryStability(
            candidate_pixel_count=100,
            in_gamut_pixel_count=100,
            stable_pixel_count=100,
            in_gamut_fraction=1.0,
            stable_fraction=1.0,
            variation_median=1.0,
            variation_p90=2.0,
        )
        self.assertFalse(_recovery_is_eligible(fit, stability, 10, 100))

    def test_compose_restores_validated_recovery_and_trusted_source(self) -> None:
        source_array = np.full((4, 4, 3), 80, dtype=np.uint8)
        generated_array = np.zeros_like(source_array)
        recovered_array = source_array.copy()
        recovered_array[1, 1] = (12, 34, 56)

        validated = np.zeros((4, 4), dtype=np.uint8)
        validated[1, 1] = 255
        trusted = np.zeros((4, 4), dtype=np.uint8)
        trusted[2, 2] = 255
        repair = LegacyBannerRepair(
            image=Image.fromarray(source_array),
            ai_mask=np.zeros((4, 4), dtype=np.uint8),
            report_mask=validated.copy(),
            recovered_mask=validated.copy(),
            structural_mask=np.zeros((4, 4), dtype=np.uint8),
            mode_used="deblend",
            recovered_fraction=1 / 16,
            recovered_image=Image.fromarray(recovered_array),
            trusted_mask=trusted,
            validated_recovered_mask=validated,
        )

        result = np.asarray(
            compose_legacy_banner_repair(
                Image.fromarray(source_array),
                Image.fromarray(generated_array),
                repair,
            )
        )
        self.assertTrue(np.array_equal(result[1, 1], recovered_array[1, 1]))
        self.assertTrue(np.array_equal(result[2, 2], source_array[2, 2]))
        self.assertTrue(np.array_equal(result[0, 0], generated_array[0, 0]))

    def test_deterministic_path_repairs_instead_of_copying_registered_foreground(self) -> None:
        source, repair = self._synthetic_promoted_repair()
        prepared = np.asarray(repair.image)

        self.assertEqual(repair.deterministic_mask[3, 4], 255)
        self.assertFalse(np.array_equal(prepared[3, 4], (20, 20, 20)))
        self.assertGreater(int(prepared[3, 4, 0]), 150)
        # A deterministic provenance pixel must be supplied by the repair, not
        # merely relabel the observed mixed source pixel.
        self.assertFalse(np.array_equal(prepared[3, 4], source[3, 4]))
        composed = np.asarray(
            compose_legacy_banner_repair(
                Image.fromarray(source),
                Image.fromarray(np.zeros_like(source)),
                repair,
            )
        )
        self.assertTrue(np.array_equal(composed[3, 4], prepared[3, 4]))

    def test_rejected_recovery_pixel_enters_executed_reconstruction_mask(self) -> None:
        _source, repair = self._synthetic_promoted_repair()

        self.assertEqual(repair.validated_recovered_mask[1, 1], 0)
        self.assertEqual(repair.deterministic_mask[1, 1], 0)
        self.assertEqual(repair.ai_mask[1, 1], 255)
        self.assertEqual(repair.generated_mask[1, 1], 255)

    def test_authorized_support_has_total_disjoint_provenance_partition(self) -> None:
        _source, repair = self._synthetic_promoted_repair()
        support = repair.authorized_support_mask > 0
        dispositions = np.stack(
            [
                repair.trusted_mask > 0,
                repair.validated_recovered_mask > 0,
                repair.deterministic_mask > 0,
                repair.generated_mask > 0,
            ]
        )

        self.assertTrue(np.all(np.sum(dispositions[:, support], axis=0) == 1))
        self.assertEqual(
            int(np.count_nonzero(repair.missing_mask)),
            int(np.count_nonzero(repair.generated_mask & repair.authorized_support_mask)),
        )

    def test_precise_mode_exposes_unhandled_panel_as_explicit_qa_failure(self) -> None:
        source = np.full((4, 5, 3), 180, dtype=np.uint8)
        operation = np.zeros((4, 5), dtype=np.uint8)
        operation[1, 2] = 255
        detection = Detection(
            "baozimh_top_left", 1.0, (0, 0, 5, 4), operation
        )
        empty = np.zeros((4, 5), dtype=bool)
        assets = LegacyBannerAssets(
            panel_shape=np.ones((4, 5), dtype=np.float32),
            foreground_support=empty,
            foreground_core=empty,
            foreground_suffix=empty,
            foreground_offset=(0, 0),
        )
        with patch(
            "wmcleaner.legacy_banner._legacy_banner_assets", return_value=assets
        ):
            repair = prepare_legacy_banner_repair(
                Image.fromarray(source), [detection], mode="precise"
            )

        self.assertEqual(repair.generated_mask[1, 2], 255)
        self.assertEqual(repair.failed_mask[1, 2], 0)
        self.assertEqual(repair.failed_pixel_count, 19)
        self.assertIn("legacy_cleanup_incomplete", repair.qa_reasons)

    def test_colored_panel_cannot_be_inverted_as_white_overlay(self) -> None:
        clean = np.full((3, 3, 3), 245, dtype=np.uint8)
        overlay = np.array((205, 225, 238), dtype=np.float32)
        alpha = 0.75
        observed = np.rint(
            clean.astype(np.float32) * (1.0 - alpha) + overlay * alpha
        ).astype(np.uint8)
        white_inverse, _valid = _inverse_white_panel(
            observed, np.ones((3, 3), dtype=np.float32), alpha, "encoded-rgb"
        )
        colored_inverse = np.rint(
            (observed.astype(np.float32) - alpha * overlay) / (1.0 - alpha)
        ).clip(0, 255).astype(np.uint8)

        self.assertGreater(np.max(np.abs(white_inverse.astype(int) - clean)), 30)
        self.assertLessEqual(np.max(np.abs(colored_inverse.astype(int) - clean)), 2)

    def test_alignment_color_and_noise_perturbations_reject_inverse_candidate(self) -> None:
        roi = np.full((5, 5, 3), 210, dtype=np.uint8)
        shape = np.ones((5, 5), dtype=np.float32)
        candidate_mask = np.ones((5, 5), dtype=bool)

        stable, evidence = _measure_recovery_stability(
            roi,
            shape,
            0.80,
            "encoded-rgb",
            candidate_mask,
            alpha_perturbation=0.02,
            alignment_perturbation=1,
            color_perturbation=2,
            variation_threshold=8,
        )

        self.assertLess(evidence.stable_fraction, 1.0)
        self.assertEqual(int(np.count_nonzero(stable[1:4, 1:4])), 0)


if __name__ == "__main__":
    unittest.main()
