from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import resources
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .detector import Detection, _load_gray_asset


LegacyMode = Literal["auto", "coherent", "deblend", "precise", "solid"]
WorkingSpace = Literal["encoded-rgb", "linear-rgb"]


# These gates intentionally remain local to translucent legacy recovery.  The
# detector's identity thresholds and the compatible solid/coherent modes do
# not use them.  A one-step alpha perturbation matches the spacing of the
# fitted alpha grid below; a one-pixel registration neighbourhood covers the
# uncertainty of the bundled foreground alignment without silently broadening
# the repair mask.
_RECOVERY_MIN_IMPROVEMENT = 0.20
_RECOVERY_MAX_GENERATED_FRACTION = 0.35
_RECOVERY_MIN_IN_GAMUT_FRACTION = 0.85
_RECOVERY_MIN_STABILITY_FRACTION = 0.80
_RECOVERY_ALPHA_PERTURBATION = 0.01
_RECOVERY_ALIGNMENT_RADIUS = 1
_RECOVERY_MAX_VARIATION = 12
# The Astra study rejected every tested semantic candidate. Keep the repaired
# deterministic branch testable, but do not promote its inverse pixels until a
# separately calibrated semantic model and held-out acceptance exist.
_LEGACY_SEMANTIC_RECOVERY_VALIDATED = False


@dataclass(frozen=True)
class LegacyBannerFit:
    working_space: WorkingSpace
    alpha: float
    baseline_error: float
    fitted_error: float
    improvement: float
    alignment_refined: bool = False
    alpha_perturbation: float = _RECOVERY_ALPHA_PERTURBATION
    alignment_perturbation: int = _RECOVERY_ALIGNMENT_RADIUS
    in_gamut_pixel_count: int = 0
    in_gamut_fraction: float = 0.0
    stable_pixel_count: int = 0
    stability_fraction: float = 0.0
    variation_median: float | None = None
    variation_p90: float | None = None
    variation_threshold: int = _RECOVERY_MAX_VARIATION
    promotion_eligible: bool = False


@dataclass(frozen=True)
class LegacyBannerAssets:
    panel_shape: np.ndarray
    foreground_support: np.ndarray
    foreground_core: np.ndarray
    foreground_suffix: np.ndarray
    foreground_offset: tuple[int, int]


@dataclass(frozen=True)
class LegacyRecoveryStability:
    """Measured sensitivity of a fitted inverse-composite candidate.

    This is deliberately separate from :class:`LegacyBannerFit`: a good
    boundary score says that a panel model fits the page, while this record
    says how much the proposed source pixel changes under small, justified
    alpha and registration perturbations.
    """

    candidate_pixel_count: int
    in_gamut_pixel_count: int
    stable_pixel_count: int
    in_gamut_fraction: float
    stable_fraction: float
    variation_median: float | None
    variation_p90: float | None
    alpha_perturbation: float = _RECOVERY_ALPHA_PERTURBATION
    alignment_perturbation: int = _RECOVERY_ALIGNMENT_RADIUS
    color_perturbation: int = 0
    variation_threshold: int = _RECOVERY_MAX_VARIATION

    @property
    def valid_fraction(self) -> float:
        """Backward/forward-friendly alias for the in-gamut fraction."""

        return self.in_gamut_fraction


@dataclass
class LegacyBannerRepair:
    image: Image.Image
    ai_mask: np.ndarray
    report_mask: np.ndarray
    recovered_mask: np.ndarray
    structural_mask: np.ndarray
    mode_used: str
    recovered_fraction: float
    boundary_blend_mask: np.ndarray | None = None
    seam_mask: np.ndarray | None = None
    seam_blend_mask: np.ndarray | None = None
    panel_pixel_count: int = 0
    deblended_pixel_count: int = 0
    ai_mask_pixel_count: int = 0
    structural_pixel_count: int = 0
    boundary_pixel_count: int = 0
    deblended_fraction: float = 0.0
    ai_mask_fraction: float = 0.0
    boundary_fraction: float = 0.0
    fit: LegacyBannerFit | None = None
    fallback_reason: str | None = None
    recovered_image: Image.Image | None = None
    # Provenance masks.  ``recovered_mask`` is the conservative promoted subset
    # for repairs produced by this module; the candidate set is retained
    # separately so callers can audit what was considered without restoring it
    # blindly after generation.
    trusted_mask: np.ndarray | None = None
    validated_recovered_mask: np.ndarray | None = None
    deterministic_mask: np.ndarray | None = None
    missing_mask: np.ndarray | None = None
    generated_mask: np.ndarray | None = None
    authorized_support_mask: np.ndarray | None = None
    failed_mask: np.ndarray | None = None
    candidate_recovered_mask: np.ndarray | None = None
    trusted_pixel_count: int = 0
    validated_recovered_pixel_count: int = 0
    deterministic_pixel_count: int = 0
    missing_pixel_count: int = 0
    generated_pixel_count: int = 0
    failed_pixel_count: int = 0
    candidate_recovered_pixel_count: int = 0
    trusted_fraction: float = 0.0
    validated_recovered_fraction: float = 0.0
    deterministic_fraction: float = 0.0
    missing_fraction: float = 0.0
    generated_fraction: float = 0.0
    candidate_recovered_fraction: float = 0.0
    recovery_promoted: bool = False
    stability: LegacyRecoveryStability | None = None
    qa_reasons: tuple[str, ...] = ()

    @property
    def qa_flags(self) -> tuple[str, ...]:
        """Alias used by manifest/QA consumers."""

        return self.qa_reasons


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.maximum(values, 0.0) ** (1.0 / 2.4) - 0.055,
    )


@lru_cache(maxsize=16)
def _panel_shape(width: int, height: int) -> np.ndarray:
    """Return the legacy rounded-panel coverage at an aligned banner size.

    The bundled reference is 325x92, while the visible rounded panel occupies
    approximately 310x89 pixels of it.  Draw at 4x resolution so the boundary
    remains a soft alpha edge after scaling.
    """
    scale_x = width / 325.0
    scale_y = height / 92.0
    panel_w = max(1, min(width, round(310 * scale_x)))
    panel_h = max(1, min(height, round(89 * scale_y)))
    radius = max(2, round(10 * min(scale_x, scale_y)))

    supersample = 4
    canvas = Image.new("L", (width * supersample, height * supersample), 0)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (
            0,
            0,
            panel_w * supersample - 1,
            panel_h * supersample - 1,
        ),
        radius=radius * supersample,
        fill=255,
    )
    canvas = canvas.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _load_rgb_asset(name: str) -> np.ndarray:
    asset = resources.files("wmcleaner").joinpath("templates", name)
    with resources.as_file(asset) as path:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


@lru_cache(maxsize=16)
def _legacy_banner_assets(width: int, height: int) -> LegacyBannerAssets:
    """Decompose the bundled crop into panel and watermark foreground layers."""
    reference = _load_rgb_asset("baozimh_top_left.png")
    template = (_load_gray_asset("baozimh_top_left_mask.png") > 0).astype(np.uint8)
    # Cross-page foreground agreement supplies URL pixels hidden by the black
    # diagonal line in the historical detector crop.
    suffix = _load_gray_asset("baozimh_top_left_suffix_mask.png") > 0
    gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV)

    rows = np.indices(template.shape)[0]
    known_rows = ((rows >= 10) & (rows < 53)) | ((rows >= 56) & (rows < 81))
    # The detector crop contains a genuinely black diagonal comic line. Direct
    # color classification removes it without the false positives that a Hough
    # pass creates across the long, nearly horizontal URL glyphs.
    contaminant = ((gray < 150) & (hsv[:, :, 1] < 60)).astype(np.uint8)
    support = (template > 0) & known_rows & ~(contaminant > 0)
    yellow = (
        support
        & (hsv[:, :, 0] >= 7)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] >= 45)
        & (hsv[:, :, 2] >= 115)
    )
    # Every cleaned template pixel is unavailable to the fitted white-panel
    # model, including faint antialiasing around gray glyphs. Keeping only a
    # color-thresholded core makes those edges inverse-composite into dark,
    # readable watermark outlines.
    modeled_support = support
    core = yellow

    offset = (
        round(-1 * width / 325.0),
        round(-2 * height / 92.0),
    )

    def aligned(layer: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            layer.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        return cv2.warpAffine(
            resized,
            np.float32([[1, 0, offset[0]], [0, 1, offset[1]]]),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        ) > 0

    return LegacyBannerAssets(
        panel_shape=_panel_shape(width, height),
        foreground_support=aligned(modeled_support),
        foreground_core=aligned(core),
        foreground_suffix=cv2.resize(
            suffix.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0,
        foreground_offset=offset,
    )


def _to_working(encoded: np.ndarray, space: WorkingSpace) -> np.ndarray:
    normalized = encoded.astype(np.float32) / 255.0
    if space == "linear-rgb":
        return _srgb_to_linear(normalized)
    return normalized


def _from_working(values: np.ndarray, space: WorkingSpace) -> np.ndarray:
    if space == "linear-rgb":
        values = _linear_to_srgb(values)
    return np.clip(values * 255.0, 0, 255).astype(np.uint8)


def _inverse_flat_panel(
    roi: np.ndarray,
    shape: np.ndarray,
    strength: float,
    space: WorkingSpace,
    panel_rgb: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    working = _to_working(roi, space)
    alpha = np.clip(shape * float(strength), 0.0, 0.94)[..., None]
    denominator = 1.0 - alpha
    panel = _to_working(np.asarray(panel_rgb, dtype=np.uint8), space)
    recovered = (working - alpha * panel) / np.maximum(denominator, 0.06)

    # Compression can move a valid inverse a few values beyond the nominal
    # range.  Larger excursions indicate that the white-panel model does not
    # explain the pixel and it should be reconstructed locally instead.
    tolerance = 8.0 / 255.0
    valid = (
        (shape > 0.01)
        & np.all(recovered >= -tolerance, axis=2)
        & np.all(recovered <= 1.0 + tolerance, axis=2)
    )
    recovered = np.clip(recovered, 0.0, 1.0)
    encoded = _from_working(recovered, space)
    result = roi.copy()
    result[valid] = encoded[valid]
    return result, valid


def _inverse_white_panel(
    roi: np.ndarray,
    shape: np.ndarray,
    strength: float,
    space: WorkingSpace,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility inverse for the historical white-panel assumption."""

    return _inverse_flat_panel(roi, shape, strength, space, (255, 255, 255))


def _boundary_fit_score(
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    candidate_roi: np.ndarray,
    foreground_mask: np.ndarray,
) -> float:
    """Measure how much the panel creates a color/gradient seam at its edges."""
    x, y, w, h = box
    panel_w = max(1, min(w, round(310 * w / 325.0)))
    panel_h = max(1, min(h, round(89 * h / 92.0)))
    radius = max(3, round(12 * min(w / 325.0, h / 92.0)))
    rgb = image_rgb.astype(np.float32)
    restored = rgb.copy()
    restored[y:y + h, x:x + w] = candidate_roi.astype(np.float32)

    samples: list[np.ndarray] = []
    exclusions: list[np.ndarray] = []

    if y >= 2:
        xs = slice(x + radius, x + panel_w - radius)
        inside = restored[y + 2, xs]
        outside_1 = rgb[y - 1, xs]
        outside_2 = rgb[y - 2, xs]
        predicted = outside_1 + 3.0 * (outside_1 - outside_2)
        samples.append(np.mean(np.abs(inside - predicted), axis=1))
        exclusions.append(foreground_mask[2, radius:panel_w - radius])

    bottom = y + panel_h
    if bottom + 1 < rgb.shape[0] and panel_h >= 3:
        xs = slice(x + radius, x + panel_w - radius)
        inside = restored[bottom - 3, xs]
        outside_1 = rgb[bottom, xs]
        outside_2 = rgb[bottom + 1, xs]
        predicted = outside_1 + 3.0 * (outside_1 - outside_2)
        samples.append(np.mean(np.abs(inside - predicted), axis=1))
        exclusions.append(foreground_mask[panel_h - 3, radius:panel_w - radius])

    if x >= 2:
        ys = slice(y + radius, y + panel_h - radius)
        inside = restored[ys, x + 2]
        outside_1 = rgb[ys, x - 1]
        outside_2 = rgb[ys, x - 2]
        predicted = outside_1 + 3.0 * (outside_1 - outside_2)
        samples.append(np.mean(np.abs(inside - predicted), axis=1))
        exclusions.append(foreground_mask[radius:panel_h - radius, 2])

    right = x + panel_w
    if right + 1 < rgb.shape[1] and panel_w >= 3:
        ys = slice(y + radius, y + panel_h - radius)
        inside = restored[ys, right - 3]
        outside_1 = rgb[ys, right]
        outside_2 = rgb[ys, right + 1]
        predicted = outside_1 + 3.0 * (outside_1 - outside_2)
        samples.append(np.mean(np.abs(inside - predicted), axis=1))
        exclusions.append(foreground_mask[radius:panel_h - radius, panel_w - 3])

    usable: list[np.ndarray] = []
    for values, excluded in zip(samples, exclusions):
        values = values[~excluded]
        if values.size:
            usable.append(values)
    if not usable:
        return float("inf")

    values = np.concatenate(usable)
    # Trim genuine artwork edges and score the majority boundary behavior.
    cutoff = np.quantile(values, 0.80)
    trimmed = values[values <= cutoff]
    return float(np.mean(trimmed)) if trimmed.size else float("inf")


def _fit_panel_model(
    image_rgb: np.ndarray,
    detection: Detection,
) -> tuple[np.ndarray, np.ndarray, LegacyBannerFit] | None:
    x, y, w, h = detection.box
    roi = image_rgb[y:y + h, x:x + w]
    if roi.shape[:2] != (h, w):
        return None

    assets = _legacy_banner_assets(w, h)
    shape = assets.panel_shape
    foreground = detection.mask[y:y + h, x:x + w] > 0
    foreground = cv2.dilate(
        foreground.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ) > 0
    baseline_error = _boundary_fit_score(image_rgb, detection.box, roi, foreground)
    best: tuple[float, WorkingSpace, float, np.ndarray, np.ndarray] | None = None

    for space in ("encoded-rgb", "linear-rgb"):
        for alpha in np.linspace(0.45, 0.88, 44):
            candidate, valid = _inverse_white_panel(
                roi,
                shape,
                float(alpha),
                space,
            )
            score = _boundary_fit_score(image_rgb, detection.box, candidate, foreground)
            if best is None or score < best[0]:
                best = (score, space, float(alpha), candidate, valid)

    if best is None or not np.isfinite(best[0]) or not np.isfinite(baseline_error):
        return None
    score, space, alpha, candidate, valid = best
    improvement = (baseline_error - score) / max(baseline_error, 1e-6)
    if improvement < 0.03:
        return None

    fit = LegacyBannerFit(
        working_space=space,
        alpha=alpha,
        baseline_error=baseline_error,
        fitted_error=score,
        improvement=improvement,
        alignment_refined=assets.foreground_offset != (0, 0),
    )
    return candidate, valid, fit


def _shift_panel_shape(shape: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a panel alpha mask for registration-neighbourhood checks."""

    height, width = shape.shape
    return cv2.warpAffine(
        shape.astype(np.float32),
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _measure_recovery_stability(
    roi: np.ndarray,
    shape: np.ndarray,
    alpha: float,
    space: WorkingSpace,
    candidate_mask: np.ndarray,
    *,
    alpha_perturbation: float = _RECOVERY_ALPHA_PERTURBATION,
    alignment_perturbation: int = _RECOVERY_ALIGNMENT_RADIUS,
    color_perturbation: int = 0,
    variation_threshold: int = _RECOVERY_MAX_VARIATION,
) -> tuple[np.ndarray, LegacyRecoveryStability]:
    """Return the subset stable under local alpha/registration perturbations.

    The central candidate is not enough to establish source recovery: an
    almost-valid alpha can produce a plausible boundary while moving a pixel
    substantially.  Evaluate one fitted-grid step in alpha and the four
    cardinal one-pixel registration neighbours.  A pixel is promoted only if
    every candidate remains in gamut and the encoded RGB variation stays
    below the conservative threshold.
    """

    central, central_valid = _inverse_white_panel(roi, shape, alpha, space)
    candidates = [central]
    valid_masks = [central_valid]

    for delta in (-alpha_perturbation, alpha_perturbation):
        candidate, valid = _inverse_white_panel(
            roi,
            shape,
            float(np.clip(alpha + delta, 0.05, 0.93)),
            space,
        )
        candidates.append(candidate)
        valid_masks.append(valid)

    # Quantization/noise and even small errors in the assumed overlay color are
    # amplified by inverse compositing.  Keep these falsification checks local
    # to the compatibility model; they do not validate or activate a semantic
    # color asset.
    if color_perturbation > 0:
        for delta in (-color_perturbation, color_perturbation):
            observed = np.clip(
                roi.astype(np.int16) + delta, 0, 255
            ).astype(np.uint8)
            candidate, valid = _inverse_white_panel(
                observed, shape, alpha, space
            )
            candidates.append(candidate)
            valid_masks.append(valid)

            panel_level = int(np.clip(255 + delta, 0, 255))
            candidate, valid = _inverse_flat_panel(
                roi,
                shape,
                alpha,
                space,
                (panel_level, panel_level, panel_level),
            )
            candidates.append(candidate)
            valid_masks.append(valid)

    for dx, dy in (
        (-alignment_perturbation, 0),
        (alignment_perturbation, 0),
        (0, -alignment_perturbation),
        (0, alignment_perturbation),
    ):
        candidate, valid = _inverse_white_panel(
            roi,
            _shift_panel_shape(shape, dx, dy),
            alpha,
            space,
        )
        candidates.append(candidate)
        valid_masks.append(valid)

    candidate_mask = candidate_mask.astype(bool)
    valid_domain = candidate_mask & np.logical_and.reduce(valid_masks)
    variation = np.max(
        np.abs(
            np.stack(
                [item.astype(np.int16) for item in candidates],
                axis=0,
            )
            - central.astype(np.int16)[None, ...]
        ),
        axis=(0, 3),
    )
    stable = valid_domain & (variation <= variation_threshold)
    candidate_pixels = int(np.count_nonzero(candidate_mask))
    in_gamut_pixels = int(np.count_nonzero(valid_domain))
    stable_pixels = int(np.count_nonzero(stable))
    if in_gamut_pixels:
        stable_variation = variation[valid_domain]
        variation_median: float | None = float(np.median(stable_variation))
        variation_p90: float | None = float(np.quantile(stable_variation, 0.90))
    else:
        variation_median = None
        variation_p90 = None

    stability = LegacyRecoveryStability(
        candidate_pixel_count=candidate_pixels,
        in_gamut_pixel_count=in_gamut_pixels,
        stable_pixel_count=stable_pixels,
        in_gamut_fraction=in_gamut_pixels / max(candidate_pixels, 1),
        stable_fraction=stable_pixels / max(candidate_pixels, 1),
        variation_median=variation_median,
        variation_p90=variation_p90,
        alpha_perturbation=alpha_perturbation,
        alignment_perturbation=alignment_perturbation,
        color_perturbation=color_perturbation,
        variation_threshold=variation_threshold,
    )
    return (stable.astype(np.uint8) * 255), stability


def _recovery_is_eligible(
    fit: LegacyBannerFit,
    stability: LegacyRecoveryStability,
    generated_pixels: int,
    panel_pixels: int,
) -> bool:
    """Apply conservative fit, stability, and localized-area promotion gates."""

    generated_fraction = generated_pixels / max(panel_pixels, 1)
    return bool(
        fit.improvement >= _RECOVERY_MIN_IMPROVEMENT
        and stability.in_gamut_fraction >= _RECOVERY_MIN_IN_GAMUT_FRACTION
        and stability.stable_fraction >= _RECOVERY_MIN_STABILITY_FRACTION
        and generated_fraction <= _RECOVERY_MAX_GENERATED_FRACTION
    )


def _legacy_local_repair_masks(
    candidate: np.ndarray,
    valid: np.ndarray,
    assets: LegacyBannerAssets,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split exact deblend from compact deterministic and AI reconstruction.

    The cleaned reference layer contains the gray title/URL and yellow icon
    strokes, but not the historical comic-line contaminant in the detector
    template. A three-pixel title expansion and one-pixel URL expansion cover
    anti-aliased edges without turning the banner rectangle into missing
    content. The compact URL layer is sent to AI after deterministic prefill
    because stroke-only interpolation retains readable letter echoes; the gray
    title uses the prefill directly. AI also receives clipped inversions and
    the opaque yellow cores.
    """
    panel = assets.panel_shape > 0.02
    rows = np.indices(panel.shape)[0]
    title_support = assets.foreground_support & (
        rows < round(48 * panel.shape[0] / 92)
    )
    url_support = assets.foreground_support & ~title_support
    title_mask = cv2.dilate(
        title_support.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ) > 0
    url_mask = cv2.dilate(
        url_support.astype(np.uint8),
        np.ones((3, 1), np.uint8),
        iterations=1,
    ) > 0
    url_mask |= cv2.dilate(
        assets.foreground_suffix.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0
    localized_foreground = title_mask | url_mask
    suffix_gap = np.zeros_like(panel)
    suffix_gap[
        round(52 * panel.shape[0] / 92):round(85 * panel.shape[0] / 92),
        round(200 * panel.shape[1] / 325):round(285 * panel.shape[1] / 325),
    ] = True
    localized_foreground |= suffix_gap
    yellow_core = cv2.dilate(
        assets.foreground_core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ) > 0
    ai_mask = panel & (~valid | yellow_core | url_mask)
    deterministic = panel & localized_foreground & ~ai_mask
    deblended = valid & panel & ~localized_foreground & ~ai_mask
    return deblended, deterministic, ai_mask


def _localized_deblend_prefill(
    candidate: np.ndarray,
    deblended: np.ndarray,
    panel: np.ndarray,
) -> np.ndarray:
    """Prefill unresolved pixels from exact deblend boundary values."""
    unresolved = panel & ~deblended
    localized = cv2.cvtColor(
        cv2.inpaint(
            cv2.cvtColor(candidate, cv2.COLOR_RGB2BGR),
            unresolved.astype(np.uint8) * 255,
            3,
            cv2.INPAINT_TELEA,
        ),
        cv2.COLOR_BGR2RGB,
    )
    result = candidate.copy()
    result[unresolved] = localized[unresolved]
    result[deblended] = candidate[deblended]
    return result


def _structural_pixel_mask(
    candidate: np.ndarray,
    deblended: np.ndarray,
) -> np.ndarray:
    """Identify observable source-derived structure within exact deblend pixels."""
    gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(candidate, cv2.COLOR_RGB2HSV)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    structural = deblended & (
        ((hsv[:, :, 1] > 25) & (gradient > 12))
        | (gray < 75)
        | (gradient > 65)
    )
    return structural.astype(np.uint8)


def _boundary_polish(
    valid: np.ndarray,
    detected_foreground: np.ndarray,
    assets: LegacyBannerAssets,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the clean box exterior plus a validated inner panel ring.

    The coherent full-region reconstruction remains untouched in the interior.
    Pixels between the rounded panel and the rectangular AI-box edge are clean
    source and are restored exactly. The translucent panel boundary uses
    inverse-composited source pixels, fading toward the generated interior.
    """
    panel = assets.panel_shape > 0.02
    scale = min(panel.shape[1] / 325.0, panel.shape[0] / 92.0)
    width = max(3, round(8 * scale))
    distance = cv2.distanceTransform(panel.astype(np.uint8), cv2.DIST_L2, 5)
    foreground = cv2.dilate(
        (detected_foreground > 0).astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=1,
    ) > 0
    ring = panel & valid & ~foreground & (distance <= width)
    exterior = ~panel & ~foreground
    blend = np.zeros(panel.shape, dtype=np.float32)
    blend[exterior] = 1.0
    blend[ring] = np.clip(
        (width + 1.0 - distance[ring]) / width,
        0.0,
        1.0,
    )
    return ring, blend


def _boundary_seam_mask(
    image_shape: tuple[int, int],
    detection: Detection,
    *,
    half_width: int | None = None,
) -> np.ndarray:
    """Mask a rounded strip crossing every edge of the legacy panel.

    The first coherent reconstruction ends at the historical detector box. A
    focused second pass needs clean pixels on both sides of the old panel edge
    so it can continue artwork across that edge instead of preserving a faint
    rectangular cutoff. Earlier versions covered only the bottom/right sides
    and excluded the rounded corners; real pages consequently retained the
    left edge or an entire rounded outline whenever the panel fit fell back.
    """
    image_h, image_w = image_shape
    x, y, w, h = detection.box
    scale = min(w / 325.0, h / 92.0)
    half_width = max(
        4,
        round(20 * scale) if half_width is None else int(half_width),
    )

    panel = np.zeros((image_h, image_w), dtype=np.uint8)
    x2 = min(image_w, x + w)
    y2 = min(image_h, y + h)
    if x >= x2 or y >= y2:
        return panel
    local = (_panel_shape(w, h) > 0.02).astype(np.uint8)
    panel[y:y2, x:x2] = local[:y2 - y, :x2 - x]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * half_width + 1, 2 * half_width + 1),
    )
    outer = cv2.dilate(panel, kernel) > 0
    inner = cv2.erode(panel, kernel) > 0
    return (outer & ~inner).astype(np.uint8) * 255


def _solid_legacy_mask(
    image_rgb: np.ndarray,
    detection: Detection,
) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    out = np.zeros(gray.shape, dtype=np.uint8)
    x, y, w, h = detection.box
    x2 = min(image_rgb.shape[1], x + w)
    y2 = min(image_rgb.shape[0], y + h)
    local = np.full((y2 - y, x2 - x), 255, dtype=np.uint8)
    core = detection.mask[y:y2, x:x2] > 0
    dark = ((gray[y:y2, x:x2] < 92) & ~core).astype(np.uint8)
    protected = cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=1) > 0
    protected &= ~core
    local[protected] = 0
    out[y:y2, x:x2] = local
    return out


def prepare_legacy_banner_repair(
    image: Image.Image,
    detections: list[Detection],
    *,
    mode: LegacyMode = "auto",
) -> LegacyBannerRepair:
    if mode not in {"auto", "coherent", "deblend", "precise", "solid"}:
        raise ValueError(f"Unsupported legacy banner mode: {mode}")

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h_img, w_img = rgb.shape[:2]
    deblended_rgb = rgb.copy()
    prepared_rgb = rgb.copy()
    ai_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    report_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    recovered_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    validated_recovered_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    deterministic_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    missing_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    generated_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    authorized_support_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    failed_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    trusted_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    structural_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    boundary_blend = np.zeros((h_img, w_img), dtype=np.float32)
    seam_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    seam_feather_sigma = 2.5
    fits: list[LegacyBannerFit] = []
    stability_records: list[LegacyRecoveryStability] = []
    fallback_reasons: list[str] = []
    qa_reasons: set[str] = set()
    recovery_promoted = False
    panel_pixels = 0

    legacy = [d for d in detections if d.name == "baozimh_top_left"]
    for detection in legacy:
        x, y, w, h = detection.box
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x >= x2 or y >= y2:
            qa_reasons.update(("weak_overlay_fit", "unstable_recovery"))
            continue
        local_h = y2 - y
        local_w = x2 - x
        assets = _legacy_banner_assets(w, h)
        local_panel = assets.panel_shape[:local_h, :local_w] > 0.02
        panel_pixels += int(np.count_nonzero(local_panel))
        panel_global = np.zeros((h_img, w_img), dtype=np.uint8)
        panel_global[y:y2, x:x2][local_panel] = 255
        authorized_support_mask = cv2.bitwise_or(
            authorized_support_mask, panel_global
        )

        # Pixels outside the rounded panel but inside the detector crop are
        # clean source context.  Keep them available for the final compose;
        # generated/seam operations are removed from this mask below.
        local_trusted = (~local_panel).astype(np.uint8) * 255
        trusted_mask[y:y2, x:x2] = cv2.bitwise_or(
            trusted_mask[y:y2, x:x2], local_trusted
        )

        if mode == "coherent":
            # A banner spanning a panel/background boundary must be redrawn as
            # one region. Preserving dark lines inside the historical box can
            # preserve the box itself, especially for top-right placements.
            pad = max(4, round(8 * min(w / 325.0, h / 92.0)))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w_img, x + w + pad)
            y1 = min(h_img, y + h + pad)
            ai_mask[y0:y1, x0:x1] = 255
            report_mask[y0:y1, x0:x1] = 255
            generated_mask[y0:y1, x0:x1] = 255
            missing_mask = cv2.bitwise_or(missing_mask, panel_global)
            continue
        if mode == "precise":
            ai_mask = cv2.bitwise_or(ai_mask, detection.mask)
            report_mask = cv2.bitwise_or(report_mask, detection.mask)
            generated_mask = cv2.bitwise_or(generated_mask, detection.mask)
            missing_mask = cv2.bitwise_or(missing_mask, panel_global)
            continue

        if mode in {"auto", "solid"}:
            solid = _solid_legacy_mask(rgb, detection)
            ai_mask = cv2.bitwise_or(ai_mask, solid)
            report_mask = cv2.bitwise_or(report_mask, solid)
            generated_mask = cv2.bitwise_or(generated_mask, solid)
            missing_mask = cv2.bitwise_or(missing_mask, panel_global)
            if mode == "solid":
                continue

            # Auto mode always gets a focused rounded-perimeter pass, even if
            # empirical panel inversion cannot improve this particular page.
            # The coherent first pass handles the interior; this second mask is
            # solely responsible for removing every visible panel boundary.
            seam_half_width = max(8, round(20 * min(w / 325.0, h / 92.0)))
            seam_mask = cv2.bitwise_or(
                seam_mask,
                _boundary_seam_mask(
                    (h_img, w_img),
                    detection,
                    half_width=seam_half_width,
                ),
            )
            seam_feather_sigma = max(
                seam_feather_sigma,
                seam_half_width / 4.0,
            )
            report_mask = cv2.bitwise_or(report_mask, seam_mask)

            fitted = _fit_panel_model(rgb, detection)
            if fitted is None:
                fallback_reasons.append("panel-fit-did-not-improve-boundary-polish-skipped")
                qa_reasons.update(("weak_overlay_fit", "unstable_recovery"))
                continue

            candidate, valid_local, fit = fitted
            local_candidate = candidate[:local_h, :local_w]
            local_assets = LegacyBannerAssets(
                panel_shape=assets.panel_shape[:local_h, :local_w],
                foreground_support=assets.foreground_support[:local_h, :local_w],
                foreground_core=assets.foreground_core[:local_h, :local_w],
                foreground_suffix=assets.foreground_suffix[:local_h, :local_w],
                foreground_offset=assets.foreground_offset,
            )
            local_valid = valid_local[:local_h, :local_w]
            candidate_direct, candidate_deterministic, candidate_ai = (
                _legacy_local_repair_masks(
                    local_candidate,
                    local_valid,
                    local_assets,
                )
            )
            stable_local, stability = _measure_recovery_stability(
                rgb[y:y2, x:x2],
                local_assets.panel_shape,
                fit.alpha,
                fit.working_space,
                candidate_direct,
            )
            stability_records.append(stability)
            candidate_ai_pixels = int(np.count_nonzero(candidate_ai))
            local_panel_pixels = int(np.count_nonzero(local_panel))
            eligible = _recovery_is_eligible(
                fit,
                stability,
                candidate_ai_pixels,
                local_panel_pixels,
            )
            fit = replace(
                fit,
                in_gamut_pixel_count=stability.in_gamut_pixel_count,
                in_gamut_fraction=stability.in_gamut_fraction,
                stable_pixel_count=stability.stable_pixel_count,
                stability_fraction=stability.stable_fraction,
                variation_median=stability.variation_median,
                variation_p90=stability.variation_p90,
                promotion_eligible=eligible,
            )
            fits.append(fit)
            if not eligible:
                if fit.improvement < _RECOVERY_MIN_IMPROVEMENT:
                    qa_reasons.update(("weak_overlay_fit", "unstable_recovery"))
                elif (
                    stability.in_gamut_fraction < _RECOVERY_MIN_IN_GAMUT_FRACTION
                    or stability.stable_fraction < _RECOVERY_MIN_STABILITY_FRACTION
                ):
                    qa_reasons.add("unstable_recovery")
                if candidate_ai_pixels / max(local_panel_pixels, 1) > _RECOVERY_MAX_GENERATED_FRACTION:
                    qa_reasons.add("excessive_generated_area")
                qa_reasons.add("recovery_model_unvalidated")
                fallback_reasons.append("recovery-model-unvalidated")
                # Keep auto's compatible solid route.  In particular, do not
                # let a weak inverse candidate redraw a boundary merely because
                # it improved the aggregate score by a few percent.
                continue

            recovery_promoted = True
            # Auto deliberately remains the known-good solid-polished route;
            # only its narrow boundary ring is eligible for source recovery.
            ring, local_blend = _boundary_polish(
                local_valid,
                detection.mask[y:y2, x:x2],
                local_assets,
            )
            # Auto keeps the established narrow perimeter promotion.  A full
            # candidate promotion is unsafe here: stable inverse pixels can
            # still contain foreground/template mismatch and would restore a
            # panel-shaped contaminant over artwork.
            promoted_ring = ring & (stable_local > 0)
            local_recovered = deblended_rgb[y:y2, x:x2]
            local_recovered[promoted_ring] = local_candidate[promoted_ring]
            validated_recovered_mask[y:y2, x:x2][promoted_ring] = 255
            boundary_blend[y:y2, x:x2] = np.maximum(
                boundary_blend[y:y2, x:x2],
                local_blend,
            )
            structural = _structural_pixel_mask(local_candidate, promoted_ring)
            structural_mask[y:y2, x:x2][structural > 0] = 255
            report_mask[y:y2, x:x2][promoted_ring] = 255
            continue

        # Explicit ``deblend`` keeps the experimental recovery-first path
        # available without making it the visual-quality default.
        fitted = _fit_panel_model(rgb, detection)
        if fitted is None:
            ai_mask = cv2.bitwise_or(ai_mask, detection.mask)
            report_mask = cv2.bitwise_or(report_mask, detection.mask)
            generated_mask = cv2.bitwise_or(generated_mask, detection.mask)
            missing_mask = cv2.bitwise_or(missing_mask, panel_global)
            fallback_reasons.append("panel-fit-did-not-improve")
            qa_reasons.update(("weak_overlay_fit", "unstable_recovery"))
            continue

        candidate, valid_local, fit = fitted
        local_valid = valid_local[:local_h, :local_w]
        local_candidate = candidate[:local_h, :local_w]
        panel = assets.panel_shape[:local_h, :local_w] > 0.02
        local_assets = LegacyBannerAssets(
            panel_shape=assets.panel_shape[:local_h, :local_w],
            foreground_support=assets.foreground_support[:local_h, :local_w],
            foreground_core=assets.foreground_core[:local_h, :local_w],
            foreground_suffix=assets.foreground_suffix[:local_h, :local_w],
            foreground_offset=assets.foreground_offset,
        )
        deblended, deterministic, localized_ai = _legacy_local_repair_masks(
            local_candidate,
            local_valid,
            local_assets,
        )
        stable_local, stability = _measure_recovery_stability(
            rgb[y:y2, x:x2],
            local_assets.panel_shape,
            fit.alpha,
            fit.working_space,
            deblended,
        )
        stability_records.append(stability)
        candidate_ai_pixels = int(np.count_nonzero(localized_ai))
        local_panel_pixels = int(np.count_nonzero(panel))
        eligible = _recovery_is_eligible(
            fit,
            stability,
            candidate_ai_pixels,
            local_panel_pixels,
        )
        if not _LEGACY_SEMANTIC_RECOVERY_VALIDATED:
            eligible = False
            qa_reasons.add("recovery_model_unvalidated")
        fit = replace(
            fit,
            in_gamut_pixel_count=stability.in_gamut_pixel_count,
            in_gamut_fraction=stability.in_gamut_fraction,
            stable_pixel_count=stability.stable_pixel_count,
            stability_fraction=stability.stable_fraction,
            variation_median=stability.variation_median,
            variation_p90=stability.variation_p90,
            promotion_eligible=eligible,
        )
        fits.append(fit)
        structural = _structural_pixel_mask(
            local_candidate,
            deblended,
        )
        local_deblended = deblended_rgb[y:y2, x:x2]
        local_deblended[deblended] = local_candidate[deblended]
        recovered_mask[y:y2, x:x2][deblended] = 255
        structural_mask[y:y2, x:x2][structural > 0] = 255
        if not eligible:
            if fit.improvement < _RECOVERY_MIN_IMPROVEMENT:
                qa_reasons.update(("weak_overlay_fit", "unstable_recovery"))
            elif (
                stability.in_gamut_fraction < _RECOVERY_MIN_IN_GAMUT_FRACTION
                or stability.stable_fraction < _RECOVERY_MIN_STABILITY_FRACTION
            ):
                qa_reasons.add("unstable_recovery")
            if candidate_ai_pixels / max(local_panel_pixels, 1) > _RECOVERY_MAX_GENERATED_FRACTION:
                qa_reasons.add("excessive_generated_area")
            qa_reasons.add("recovery_model_unvalidated")
            # Recovery candidates remain inspectable through recovered_mask,
            # but the explicit mode falls back to the narrow detector mask.
            ai_mask[y:y2, x:x2] = cv2.bitwise_or(
                ai_mask[y:y2, x:x2],
                detection.mask[y:y2, x:x2],
            )
            generated_mask[y:y2, x:x2] = cv2.bitwise_or(
                generated_mask[y:y2, x:x2],
                detection.mask[y:y2, x:x2],
            )
            report_mask[y:y2, x:x2] = cv2.bitwise_or(
                report_mask[y:y2, x:x2],
                detection.mask[y:y2, x:x2],
            )
            missing_mask[y:y2, x:x2][panel] = 255
            fallback_reasons.append("recovery-model-unvalidated")
            continue

        recovery_promoted = True
        stable = (stable_local > 0) & deblended
        rejected = deblended & ~stable
        reconstruction = localized_ai | rejected
        ai_mask[y:y2, x:x2][reconstruction] = 255
        generated_mask[y:y2, x:x2][reconstruction] = 255
        deterministic_mask[y:y2, x:x2][deterministic] = 255
        validated_recovered_mask[y:y2, x:x2][stable] = 255
        report_mask[y:y2, x:x2][stable] = 255
        report_mask[y:y2, x:x2][deterministic] = 255
        report_mask[y:y2, x:x2][reconstruction] = 255
        # Keep only validated recovery and deterministic foreground cleanup in
        # the model input.  The former implementation prefilling every
        # in-gamut candidate made unstable pixels look recovered even though
        # they were not safe to restore, and those changes were absent from
        # the operation report.  Unresolved holes remain source pixels until
        # the explicitly localized AI mask handles them.
        local_prepared = rgb[y:y2, x:x2].copy()
        deterministic_repair = _localized_deblend_prefill(
            local_candidate,
            stable,
            panel,
        )
        local_prepared[stable] = local_candidate[stable]
        local_prepared[deterministic] = deterministic_repair[deterministic]
        local_prepared[reconstruction] = deterministic_repair[reconstruction]
        prepared_rgb[y:y2, x:x2] = local_prepared
        missing_mask[y:y2, x:x2][reconstruction] = 255

    # Convert operation masks into a coherent final-disposition partition over
    # the complete authorized panel support.  Precise mode deliberately does
    # not authorize broad panel reconstruction, so any remaining mixed panel
    # pixels are explicit QA failures.  Other compatibility modes send every
    # unresolved support pixel to an actual backend mask.
    support = authorized_support_mask > 0
    validated = (validated_recovered_mask > 0) & support
    deterministic = (deterministic_mask > 0) & support & ~validated
    trusted_source = (trusted_mask > 0) & ~validated & ~deterministic
    generated = (generated_mask > 0) & ~validated & ~deterministic
    trusted_source &= ~generated
    trusted = trusted_source & support
    unresolved = support & ~validated & ~deterministic & ~trusted & ~generated
    if np.any(unresolved):
        if mode == "precise":
            failed_mask[unresolved] = 255
            qa_reasons.add("legacy_cleanup_incomplete")
        else:
            ai_mask[unresolved] = 255
            generated[unresolved] = True
            report_mask[unresolved] = 255

    validated_recovered_mask = validated.astype(np.uint8) * 255
    deterministic_mask = deterministic.astype(np.uint8) * 255
    trusted_mask = trusted_source.astype(np.uint8) * 255
    generated_mask = generated.astype(np.uint8) * 255
    failed = failed_mask > 0
    missing_mask = ((generated & support) | failed).astype(np.uint8) * 255

    # Preserve the full in-gamut candidate for diagnostics, but expose and
    # count only the stability-gated subset as ``recovered_mask``.  This keeps
    # compose and existing callers aligned: anything named recovered is safe
    # to restore, while candidate_recovered_mask shows the rejected remainder.
    candidate_recovered_mask = recovered_mask
    candidate_deblended_pixels = int(np.count_nonzero(candidate_recovered_mask))
    recovered_mask = validated_recovered_mask.copy()
    deblended_pixels = int(np.count_nonzero(recovered_mask))
    ai_mask_pixels = int(np.count_nonzero(ai_mask))
    structural_pixels = int(np.count_nonzero(structural_mask))
    validated_pixels = int(np.count_nonzero(validated_recovered_mask))
    deterministic_pixels = int(np.count_nonzero(deterministic_mask))
    missing_pixels = int(np.count_nonzero(missing_mask))
    generated_pixels = int(np.count_nonzero(generated_mask))
    failed_pixels = int(np.count_nonzero(failed_mask))
    trusted_mask = cv2.bitwise_and(
        trusted_mask,
        cv2.bitwise_not(
            cv2.bitwise_or(
                cv2.bitwise_or(generated_mask, seam_mask),
                (boundary_blend > 0).astype(np.uint8) * 255,
            )
        ),
    )
    trusted_pixels = int(np.count_nonzero(trusted_mask))
    boundary_pixels = int(
        np.count_nonzero((boundary_blend > 0.0) | (seam_mask > 0))
    )
    seam_blend = cv2.GaussianBlur(
        (seam_mask > 0).astype(np.float32),
        (0, 0),
        sigmaX=seam_feather_sigma,
        sigmaY=seam_feather_sigma,
    )
    deblended_fraction = deblended_pixels / panel_pixels if panel_pixels else 0.0
    ai_mask_fraction = ai_mask_pixels / panel_pixels if panel_pixels else 0.0
    boundary_fraction = boundary_pixels / panel_pixels if panel_pixels else 0.0
    validated_fraction = validated_pixels / panel_pixels if panel_pixels else 0.0
    deterministic_fraction = deterministic_pixels / panel_pixels if panel_pixels else 0.0
    missing_fraction = missing_pixels / panel_pixels if panel_pixels else 0.0
    generated_fraction = generated_pixels / panel_pixels if panel_pixels else 0.0
    trusted_fraction = trusted_pixels / panel_pixels if panel_pixels else 0.0

    if mode == "auto":
        mode_used = "solid-polished" if fits else "solid"
    elif mode == "deblend":
        mode_used = "deblend" if recovery_promoted else "broad-fallback"
    else:
        mode_used = mode
    fit = min(fits, key=lambda value: value.fitted_error) if fits else None
    reason = ",".join(sorted(set(fallback_reasons))) or None
    stability = (
        min(stability_records, key=lambda value: value.stable_fraction)
        if stability_records
        else None
    )
    # Every operation that changes pixels is part of report_mask.  This final
    # union also protects callers that inspect/modify the individual masks.
    report_mask = cv2.bitwise_or(report_mask, validated_recovered_mask)
    report_mask = cv2.bitwise_or(report_mask, deterministic_mask)
    report_mask = cv2.bitwise_or(report_mask, generated_mask)
    report_mask = cv2.bitwise_or(report_mask, seam_mask)
    report_mask = cv2.bitwise_or(
        report_mask,
        (boundary_blend > 0).astype(np.uint8) * 255,
    )
    return LegacyBannerRepair(
        image=(
            Image.fromarray(prepared_rgb)
            if mode == "deblend" and recovery_promoted
            else image.copy()
        ),
        ai_mask=ai_mask,
        report_mask=report_mask,
        recovered_mask=recovered_mask,
        structural_mask=structural_mask,
        mode_used=mode_used,
        recovered_fraction=(
            boundary_fraction if mode == "auto" else deblended_fraction
        ),
        boundary_blend_mask=boundary_blend,
        seam_mask=seam_mask,
        seam_blend_mask=seam_blend,
        panel_pixel_count=panel_pixels,
        deblended_pixel_count=deblended_pixels,
        ai_mask_pixel_count=ai_mask_pixels,
        structural_pixel_count=structural_pixels,
        boundary_pixel_count=boundary_pixels,
        deblended_fraction=deblended_fraction,
        ai_mask_fraction=ai_mask_fraction,
        boundary_fraction=boundary_fraction,
        fit=fit,
        fallback_reason=reason,
        recovered_image=(
            Image.fromarray(deblended_rgb)
            if mode == "deblend" and recovery_promoted
            else (
                Image.fromarray(deblended_rgb)
                if mode == "auto" and fits
                else None
            )
        ),
        trusted_mask=trusted_mask,
        validated_recovered_mask=validated_recovered_mask,
        deterministic_mask=deterministic_mask,
        missing_mask=missing_mask,
        generated_mask=generated_mask,
        authorized_support_mask=authorized_support_mask,
        failed_mask=failed_mask,
        candidate_recovered_mask=candidate_recovered_mask,
        trusted_pixel_count=trusted_pixels,
        validated_recovered_pixel_count=validated_pixels,
        deterministic_pixel_count=deterministic_pixels,
        missing_pixel_count=missing_pixels,
        generated_pixel_count=generated_pixels,
        failed_pixel_count=failed_pixels,
        candidate_recovered_pixel_count=candidate_deblended_pixels,
        trusted_fraction=trusted_fraction,
        validated_recovered_fraction=validated_fraction,
        deterministic_fraction=deterministic_fraction,
        missing_fraction=missing_fraction,
        generated_fraction=generated_fraction,
        candidate_recovered_fraction=(
            candidate_deblended_pixels / panel_pixels if panel_pixels else 0.0
        ),
        recovery_promoted=recovery_promoted,
        stability=stability,
        qa_reasons=tuple(sorted(qa_reasons)),
    )


def compose_legacy_banner_repair(
    source: Image.Image,
    generated: Image.Image,
    repair: LegacyBannerRepair,
) -> Image.Image:
    """Compose generated pixels while restoring validated source evidence.

    Backends generally preserve unmasked pixels, but that is not a sufficient
    contract for a recovery-first route: a model or a later seam pass can
    still overwrite pixels that were explicitly trusted.  Restore validated
    inverse-composite pixels and trusted source context as the final operation.
    ``validated_recovered_mask`` is intentionally preferred to the historical
    ``recovered_mask`` candidate mask; a manually constructed legacy repair
    without the new field retains the old behavior as a compatibility fallback.
    """
    result = np.asarray(generated.convert("RGB"), dtype=np.uint8).copy()
    source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    if (
        repair.mode_used == "solid-polished"
        and repair.boundary_blend_mask is not None
        and repair.recovered_image is not None
    ):
        recovered = np.asarray(repair.recovered_image.convert("RGB"), dtype=np.uint8)
        alpha = repair.boundary_blend_mask[..., None]
        result = np.clip(
            result.astype(np.float32) * (1.0 - alpha)
            + recovered.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)

    if repair.recovered_image is not None:
        recovered = np.asarray(repair.recovered_image.convert("RGB"), dtype=np.uint8)
        validated = repair.validated_recovered_mask
        if validated is None:
            # Compatibility for callers that instantiate the old dataclass
            # shape directly.  Repairs produced by this module always provide
            # an explicit validated mask.
            validated = repair.recovered_mask
        if validated is not None:
            exact = validated > 0
            result[exact] = recovered[exact]

    trusted = repair.trusted_mask
    if trusted is not None:
        result[trusted > 0] = source_rgb[trusted > 0]
    deterministic = repair.deterministic_mask
    if deterministic is not None:
        repaired = np.asarray(repair.image.convert("RGB"), dtype=np.uint8)
        result[deterministic > 0] = repaired[deterministic > 0]
    return Image.fromarray(result)
