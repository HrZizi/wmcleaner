"""Recovery-first evidence and routing for individual watermark overlays.

The cleaner historically made several page-level decisions from brightness,
location, or an aggregate overlap score.  This module keeps those signals as
evidence, but gives each detection its own plan and explicit masks.  The
implementation is deliberately conservative: a surface fit is only promoted
when it predicts held-out boundary samples, and structure which crosses the
overlay boundary is trusted before any reconstruction is considered.

The data classes in this module contain no model-specific state.  They are
therefore useful both to the image cleaner and to callers which want to record
why an output was accepted or sent for review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from .detector import _observed_logo_icon_box


def _as_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    return rgb


def _empty_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def _binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask, dtype=np.uint8)
    if value.shape != shape:
        resized = cv2.resize(value, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        value = resized
    return (value > 0).astype(np.uint8) * 255


def _box_slices(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    x, y, w, h = (int(v) for v in box)
    ih, iw = shape
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(iw, x + max(0, w))
    y1 = min(ih, y + max(0, h))
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def _robust_fit(
    coordinates: np.ndarray,
    values: np.ndarray,
    model: str,
    *,
    robust_constant: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit a constant or affine RGB surface with a few robust reweighting passes."""
    if len(coordinates) == 0 or len(values) != len(coordinates):
        return None
    x = coordinates.astype(np.float64, copy=False)
    if model == "constant":
        design = np.ones((len(x), 1), dtype=np.float64)
        if robust_constant:
            # The median minimizes absolute prediction error. IRLS started at
            # the mean can stop before reweighting when a constant majority
            # makes the residual MAD zero, leaving minority artwork in the fill.
            return np.median(values, axis=0, keepdims=True), design
    elif model == "affine":
        design = np.column_stack((np.ones(len(x)), x[:, 0], x[:, 1]))
    else:
        raise ValueError(f"Unsupported surface model: {model}")

    target = values.astype(np.float64, copy=False)
    weights = np.ones(len(x), dtype=np.float64)
    coefficients: np.ndarray | None = None
    for _ in range(3):
        weighted_design = design * weights[:, None]
        weighted_target = target * weights[:, None]
        try:
            coefficients = np.linalg.lstsq(
                weighted_design,
                weighted_target,
                rcond=None,
            )[0]
        except np.linalg.LinAlgError:
            return None
        error = np.mean(np.abs(design @ coefficients - target), axis=1)
        scale = float(np.median(np.abs(error - np.median(error)))) * 1.4826
        if not np.isfinite(scale) or scale < 0.25:
            break
        weights = 1.0 / np.maximum(1.0, error / (2.5 * scale))
    if coefficients is None:
        return None
    return coefficients, design


def _surface_prediction(
    coefficients: tuple[float, ...],
    model: str,
    coordinates: np.ndarray,
) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64).reshape(-1, 3)
    if model == "constant":
        return np.broadcast_to(values[0], (len(coordinates), 3)).copy()
    design = np.column_stack((np.ones(len(coordinates)), coordinates[:, 0], coordinates[:, 1]))
    return design @ values


def _surface_coordinates(
    yy: np.ndarray,
    xx: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    x, y, w, h = box
    return np.column_stack(
        (
            (xx.astype(np.float64) - float(x + max(1, w) / 2.0)) / max(1.0, w / 2.0),
            (yy.astype(np.float64) - float(y + max(1, h) / 2.0)) / max(1.0, h / 2.0),
        )
    )


def _mean_absolute_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(actual) == 0:
        return float("inf")
    return float(np.mean(np.abs(predicted.astype(np.float64) - actual.astype(np.float64))))


@dataclass(frozen=True)
class SurfaceEvidence:
    model: str | None
    valid: bool
    sample_count: int
    pale_fraction: float
    constant_holdout_error: float
    affine_holdout_error: float
    constant_seam_error: float
    affine_seam_error: float
    complexity_gain: float
    coefficients: tuple[float, ...] = ()
    origin: tuple[int, int, int, int] | None = None
    rejection_reason: str | None = None

    @property
    def holdout_error(self) -> float:
        if self.model == "affine":
            return self.affine_holdout_error
        return self.constant_holdout_error

    @property
    def seam_error(self) -> float:
        if self.model == "affine":
            return self.affine_seam_error
        return self.constant_seam_error


@dataclass(frozen=True)
class IdentityEvidence:
    family: str
    variant: str
    confidence: float
    variant_confidence: float
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegistrationEvidence:
    confidence: float
    offset: tuple[int, int] = (0, 0)
    source: str = "detector-aligned"
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneEvidence:
    scene_class: str
    confidence: float
    structure_pixels: int
    structure_fraction: float
    geometry_kind: str | None = None
    geometry_supported: bool = False
    geometry_support: float = 0.0
    geometry_candidates: int = 0
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecoveryEvidence:
    confidence: float
    recoverable_pixels: int
    deterministic_pixels: int
    localized_pixels: int
    broad_pixels: int
    fit_valid: bool = False
    fit_stability: float = 0.0
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    generation_scope: str
    reason: str
    compatibility_mode: str | None = None


@dataclass(frozen=True)
class ResidualEvidence:
    registered_components: int = 0
    candidate_pixels: int = 0
    qualifying_components: int = 0
    max_coverage: float = 0.0
    should_flag: bool = False
    reason: str | None = None


@dataclass
class OverlayPlan:
    """Independent cleanup plan for one detection."""

    overlay_id: str
    detection_index: int
    name: str
    box: tuple[int, int, int, int]
    identity: IdentityEvidence
    registration: RegistrationEvidence
    scene: SceneEvidence
    recovery: RecoveryEvidence
    routing: RoutingDecision
    surface: SurfaceEvidence
    trusted_source_mask: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros((0, 0), dtype=np.uint8))
    recovered_mask: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros((0, 0), dtype=np.uint8))
    deterministic_mask: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros((0, 0), dtype=np.uint8))
    missing_information_mask: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros((0, 0), dtype=np.uint8))
    generated_mask: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros((0, 0), dtype=np.uint8))
    failed_mask: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros((0, 0), dtype=np.uint8))
    residual: ResidualEvidence = field(default_factory=ResidualEvidence)
    qa_reasons: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def family(self) -> str:
        return self.identity.family

    @property
    def route(self) -> str:
        return self.routing.route

    @route.setter
    def route(self, value: str) -> None:
        self.routing = RoutingDecision(
            route=value,
            generation_scope=self.routing.generation_scope,
            reason=self.routing.reason,
            compatibility_mode=self.routing.compatibility_mode,
        )

    @property
    def missing_mask(self) -> np.ndarray:
        """Compatibility alias for the missing-information mask."""
        return self.missing_information_mask

    def metadata(self) -> dict[str, Any]:
        """Return JSON-friendly evidence without embedding full image masks."""
        return {
            "overlay_id": self.overlay_id,
            "detection_index": self.detection_index,
            "name": self.name,
            "box": list(self.box),
            "identity": {
                "family": self.identity.family,
                "variant": self.identity.variant,
                "confidence": round(float(self.identity.confidence), 6),
                "variant_confidence": round(float(self.identity.variant_confidence), 6),
                "evidence": list(self.identity.evidence),
            },
            "registration": {
                "confidence": round(float(self.registration.confidence), 6),
                "offset": list(self.registration.offset),
                "source": self.registration.source,
                "evidence": list(self.registration.evidence),
            },
            "scene": {
                "class": self.scene.scene_class,
                "confidence": round(float(self.scene.confidence), 6),
                "structure_pixels": self.scene.structure_pixels,
                "structure_fraction": round(float(self.scene.structure_fraction), 6),
                "geometry": self.scene.geometry_kind,
                "geometry_supported": self.scene.geometry_supported,
                "geometry_support": round(float(self.scene.geometry_support), 6),
                "geometry_candidates": self.scene.geometry_candidates,
                "evidence": list(self.scene.evidence),
            },
            "recovery": {
                "confidence": round(float(self.recovery.confidence), 6),
                "recoverable_pixels": self.recovery.recoverable_pixels,
                "deterministic_pixels": self.recovery.deterministic_pixels,
                "localized_pixels": self.recovery.localized_pixels,
                "broad_pixels": self.recovery.broad_pixels,
                "fit_valid": self.recovery.fit_valid,
                "fit_stability": round(float(self.recovery.fit_stability), 6),
                "evidence": list(self.recovery.evidence),
            },
            "surface": {
                "model": self.surface.model,
                "valid": self.surface.valid,
                "sample_count": self.surface.sample_count,
                "pale_fraction": round(float(self.surface.pale_fraction), 6),
                "constant_holdout_error": round(float(self.surface.constant_holdout_error), 6),
                "affine_holdout_error": round(float(self.surface.affine_holdout_error), 6),
                "constant_seam_error": round(float(self.surface.constant_seam_error), 6),
                "affine_seam_error": round(float(self.surface.affine_seam_error), 6),
                "complexity_gain": round(float(self.surface.complexity_gain), 6),
                "rejection_reason": self.surface.rejection_reason,
            },
            "routing": {
                "route": self.routing.route,
                "generation_scope": self.routing.generation_scope,
                "reason": self.routing.reason,
                "compatibility_mode": self.routing.compatibility_mode,
            },
            "masks": {
                "trusted_source_pixels": int(np.count_nonzero(self.trusted_source_mask)),
                "recovered_pixels": int(np.count_nonzero(self.recovered_mask)),
                "deterministic_pixels": int(np.count_nonzero(self.deterministic_mask)),
                "missing_information_pixels": int(np.count_nonzero(self.missing_information_mask)),
                "generated_pixels": int(np.count_nonzero(self.generated_mask)),
                "failed_pixels": int(np.count_nonzero(self.failed_mask)),
            },
            "residual": {
                "registered_components": self.residual.registered_components,
                "candidate_pixels": self.residual.candidate_pixels,
                "qualifying_components": self.residual.qualifying_components,
                "max_coverage": round(float(self.residual.max_coverage), 6),
                "should_flag": self.residual.should_flag,
                "reason": self.residual.reason,
            },
            "qa_reasons": list(self.qa_reasons),
            "provenance": dict(self.provenance),
        }


@dataclass
class ResidualCheck:
    """Universal post-clean residual decision and its corrective operation mask."""

    candidate_mask: np.ndarray
    operation_mask: np.ndarray
    components: tuple[dict[str, Any], ...]
    should_flag: bool
    reasons: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return not self.should_flag

    def metadata(self) -> dict[str, Any]:
        return {
            "candidate_pixels": int(np.count_nonzero(self.candidate_mask)),
            "operation_pixels": int(np.count_nonzero(self.operation_mask)),
            "components": [dict(component) for component in self.components],
            "should_flag": self.should_flag,
            "reasons": list(self.reasons),
        }


def _ring_and_expanded_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    clipped = _box_slices(box, shape)
    if clipped is None:
        return None
    x0, y0, x1, y1 = clipped
    width = x1 - x0
    height = y1 - y0
    margin = max(8, min(48, round(0.18 * min(max(1, width), max(1, height)))))
    ex0 = max(0, x0 - margin)
    ey0 = max(0, y0 - margin)
    ex1 = min(shape[1], x1 + margin)
    ey1 = min(shape[0], y1 + margin)
    expanded = np.zeros((ey1 - ey0, ex1 - ex0), dtype=bool)
    expanded[:] = True
    expanded[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = False
    return (ex0, ey0, ex1, ey1), expanded, np.array((x0, y0, x1, y1), dtype=np.int32)


def _fit_surface_evidence(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    family: str,
) -> tuple[SurfaceEvidence, np.ndarray]:
    """Fit constant/affine surfaces and return a full-size prediction image."""
    shape = rgb.shape[:2]
    expanded_info = _ring_and_expanded_box(box, shape)
    if expanded_info is None:
        return (
            SurfaceEvidence(
                model=None,
                valid=False,
                sample_count=0,
                pale_fraction=0.0,
                constant_holdout_error=float("inf"),
                affine_holdout_error=float("inf"),
                constant_seam_error=float("inf"),
                affine_seam_error=float("inf"),
                complexity_gain=0.0,
                rejection_reason="invalid-overlay-box",
            ),
            np.zeros_like(rgb),
        )

    (ex0, ey0, ex1, ey1), ring, clipped = expanded_info
    x0, y0, x1, y1 = (int(v) for v in clipped)
    region = rgb[ey0:ey1, ex0:ex1]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    gradients = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    # Robust surface samples are pale/neutral and locally smooth.  The
    # brightness guard intentionally rejects dark artwork; a dark scene must
    # use a missing-information mask, never a pale fill.
    pale_threshold = 235 if family == "legacy" else 150
    pale = (gray >= pale_threshold) & (hsv[:, :, 1] <= 115)
    smooth = gradients <= 48.0
    sample_mask = ring & pale & smooth
    yy, xx = np.nonzero(sample_mask)
    if len(xx):
        coords = _surface_coordinates(yy + ey0, xx + ex0, box)
        values = region[yy, xx]
    else:
        coords = np.empty((0, 2), dtype=np.float64)
        values = np.empty((0, 3), dtype=np.uint8)

    if len(coords) < max(64, round(0.012 * max(1, (x1 - x0) * (y1 - y0)))):
        prediction = np.zeros_like(rgb)
        return (
            SurfaceEvidence(
                model=None,
                valid=False,
                sample_count=len(coords),
                pale_fraction=float(np.mean(pale[ring])) if np.any(ring) else 0.0,
                constant_holdout_error=float("inf"),
                affine_holdout_error=float("inf"),
                constant_seam_error=float("inf"),
                affine_seam_error=float("inf"),
                complexity_gain=0.0,
                rejection_reason="insufficient-pale-boundary-support",
            ),
            prediction,
        )

    # Cap samples deterministically so large pages do not change weighting.
    if len(coords) > 24000:
        indices = np.linspace(0, len(coords) - 1, 24000).astype(np.int64)
        coords = coords[indices]
        values = values[indices]
        local_x = xx[indices]
        local_y = yy[indices]
    else:
        local_x = xx
        local_y = yy

    side = np.full(len(coords), "interior", dtype=object)
    # The side labels are based on the expanded ring, not image location.
    side[(local_y <= max(0, y0 - ey0 + 2))] = "top"
    side[(local_y >= min(region.shape[0] - 1, y1 - ey0 - 2))] = "bottom"
    side[(local_x <= max(0, x0 - ex0 + 2))] = "left"
    side[(local_x >= min(region.shape[1] - 1, x1 - ex0 - 2))] = "right"
    side_values = np.unique(side)

    fits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in ("constant", "affine"):
        fit = _robust_fit(coords, values, model, robust_constant=family == "modern")
        if fit is not None:
            fits[model] = fit

    if "constant" not in fits or "affine" not in fits:
        prediction = np.zeros_like(rgb)
        return (
            SurfaceEvidence(
                model=None,
                valid=False,
                sample_count=len(coords),
                pale_fraction=float(np.mean(pale[ring])) if np.any(ring) else 0.0,
                constant_holdout_error=float("inf"),
                affine_holdout_error=float("inf"),
                constant_seam_error=float("inf"),
                affine_seam_error=float("inf"),
                complexity_gain=0.0,
                rejection_reason="surface-fit-failed",
            ),
            prediction,
        )

    holdout: dict[str, list[float]] = {"constant": [], "affine": []}
    for held_out in side_values:
        if held_out == "interior":
            continue
        train = side != held_out
        test = side == held_out
        if np.count_nonzero(test) < 8 or np.count_nonzero(train) < 32:
            continue
        for model in ("constant", "affine"):
            fit = _robust_fit(
                coords[train], values[train], model,
                robust_constant=family == "modern",
            )
            if fit is None:
                continue
            coefficients, _ = fit
            predicted = _surface_prediction(
                tuple(float(v) for v in coefficients.reshape(-1)),
                model,
                coords[test],
            )
            holdout[model].append(_mean_absolute_error(predicted, values[test]))

    full_coefficients: dict[str, tuple[float, ...]] = {}
    for model, (coefficients, _) in fits.items():
        full_coefficients[model] = tuple(float(v) for v in coefficients.reshape(-1))

    constant_holdout = float(np.mean(holdout["constant"])) if holdout["constant"] else float("inf")
    affine_holdout = float(np.mean(holdout["affine"])) if holdout["affine"] else float("inf")

    # Compare prediction to clean pixels immediately outside each side.  This
    # is a cheap boundary continuity proxy and catches a flat fill on a smooth
    # gradient even when the side-wise holdout error is close.
    boundary_mask = np.zeros_like(ring)
    boundary_mask[max(0, y0 - ey0 - 1):max(0, y0 - ey0), x0 - ex0:x1 - ex0] = True
    boundary_mask[min(region.shape[0], y1 - ey0):min(region.shape[0], y1 - ey0 + 1), x0 - ex0:x1 - ex0] = True
    boundary_mask[y0 - ey0:y1 - ey0, max(0, x0 - ex0 - 1):max(0, x0 - ex0)] = True
    boundary_mask[y0 - ey0:y1 - ey0, min(region.shape[1], x1 - ex0):min(region.shape[1], x1 - ex0 + 1)] = True
    by, bx = np.nonzero(boundary_mask & pale)
    boundary_coords = _surface_coordinates(by + ey0, bx + ex0, box)
    boundary_values = region[by, bx]
    seam_values: dict[str, float] = {}
    for model in ("constant", "affine"):
        predicted = _surface_prediction(full_coefficients[model], model, boundary_coords)
        seam_values[model] = _mean_absolute_error(predicted, boundary_values)

    pale_fraction = float(np.mean(pale[ring])) if np.any(ring) else 0.0
    complexity_gain = max(0.0, (constant_holdout - affine_holdout) / max(constant_holdout, 1e-6))
    # A model is promoted only when it has plausible pale support and bounded
    # held-out/seam error.  Affine wins only with a material complexity gain;
    # this keeps a nearly constant cream page on the simpler model.
    affine_wins = (
        affine_holdout <= 12.0
        and seam_values["affine"] <= 14.0
        and complexity_gain >= 0.15
        and affine_holdout + 0.50 < constant_holdout
    )
    constant_valid = constant_holdout <= 14.0 and seam_values["constant"] <= 16.0
    if family == "modern" and len(boundary_values):
        # Aggregate MAE tolerates a visible systematic seam on an otherwise
        # constant page. Require agreement with the typical local perimeter,
        # while retaining the existing holdout/affine guards for gradients.
        constant = np.asarray(full_coefficients["constant"])
        perimeter_bias = float(np.max(np.abs(np.median(boundary_values, axis=0) - constant)))
        constant_valid = constant_valid and perimeter_bias <= 2.0
    affine_valid = affine_holdout <= 14.0 and seam_values["affine"] <= 16.0
    if family not in {"modern", "legacy"}:
        constant_valid = affine_valid = False

    # A legacy translucent panel needs a genuinely pale surrounding margin;
    # accepting a small bright island in a colored scene recreates the old
    # location/brightness shortcut.  Modern pages tolerate a little more
    # color, but still require enough pale boundary evidence for a surface.
    minimum_pale_fraction = 0.80 if family == "legacy" else 0.62
    if pale_fraction < minimum_pale_fraction:
        constant_valid = affine_valid = False

    if affine_wins and affine_valid:
        selected = "affine"
    elif constant_valid:
        selected = "constant"
    elif affine_valid:
        selected = "affine"
    else:
        selected = None

    full_prediction = np.zeros_like(rgb)
    prediction_model = selected
    if prediction_model is None and full_coefficients:
        # Even a rejected surface needs a plausible reference for structure
        # analysis; the evidence remains invalid and is never applied.
        prediction_model = min(
            full_coefficients,
            key=lambda value: (
                affine_holdout if value == "affine" else constant_holdout
            ),
        )
    if prediction_model is not None:
        yy_all, xx_all = np.indices(shape)
        coords_all = _surface_coordinates(yy_all.reshape(-1), xx_all.reshape(-1), box)
        prediction_all = _surface_prediction(
            full_coefficients[prediction_model],
            prediction_model,
            coords_all,
        )
        full_prediction = np.clip(
            np.rint(prediction_all.reshape(shape[0], shape[1], 3)),
            0,
            255,
        ).astype(np.uint8)

    reason = None if selected else "boundary-fit-not-validated"
    evidence = SurfaceEvidence(
        model=selected,
        valid=selected is not None,
        sample_count=len(coords),
        pale_fraction=pale_fraction,
        constant_holdout_error=constant_holdout,
        affine_holdout_error=affine_holdout,
        constant_seam_error=seam_values["constant"],
        affine_seam_error=seam_values["affine"],
        complexity_gain=complexity_gain,
        coefficients=full_coefficients[selected] if selected else (),
        origin=tuple(int(v) for v in box),
        rejection_reason=reason,
    )
    return evidence, full_prediction


def _registered_template_mask(
    detection: Any,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = getattr(detection, "mask", None)
    if mask is None:
        return _empty_mask(shape)
    return _binary_mask(mask, shape)


def _structure_and_geometry(
    rgb: np.ndarray,
    detection: Any,
    surface: SurfaceEvidence,
    prediction: np.ndarray,
) -> tuple[np.ndarray, SceneEvidence]:
    shape = rgb.shape[:2]
    clipped = _box_slices(tuple(detection.box), shape)
    if clipped is None:
        scene = SceneEvidence("unknown", 0.0, 0, 0.0, evidence=("invalid-overlay-box",))
        return _empty_mask(shape), scene
    x0, y0, x1, y1 = clipped
    expanded_info = _ring_and_expanded_box(tuple(detection.box), shape)
    if expanded_info is None:
        scene = SceneEvidence("unknown", 0.0, 0, 0.0, evidence=("invalid-overlay-box",))
        return _empty_mask(shape), scene
    (ex0, ey0, ex1, ey1), _, _ = expanded_info
    region = rgb[ey0:ey1, ex0:ex1]
    local_prediction = prediction[ey0:ey1, ex0:ex1]
    prediction_available = bool(np.any(prediction))
    if not prediction_available:
        # A rejected surface still needs a neutral reference for structure
        # analysis.  Treating the all-zero placeholder as a prediction would
        # classify every bright pixel as connected structure and would make a
        # simple mixed boundary look like a page-sized drawing.
        fallback = np.median(region.reshape(-1, 3), axis=0).astype(np.uint8)
        local_prediction = np.broadcast_to(fallback, region.shape).copy()
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    local_gray = cv2.cvtColor(local_prediction, cv2.COLOR_RGB2GRAY)
    deviation = np.max(np.abs(region.astype(np.int16) - local_prediction.astype(np.int16)), axis=2)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    # Residual-to-surface structure, dark ink, and saturated artwork are
    # candidates.  The subsequent connectivity test is what makes them
    # trusted: isolated watermark pixels cannot authorize preservation.
    candidate = (
        ((deviation >= 9) & (gradient >= 8))
        if prediction_available
        else (gradient >= 18)
    )
    candidate |= (gradient >= 18) & (deviation >= 5)
    candidate |= (gray < 130) & (deviation >= 5)
    candidate |= hsv[:, :, 1] >= 72
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    ) > 0
    inner = np.zeros(candidate.shape, dtype=bool)
    inner[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = True
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    trusted_local = np.zeros(candidate.shape, dtype=bool)
    # Registered foreground is never treated as trusted page structure by
    # itself.  A component must have clean evidence on both sides of the box.
    template = _registered_template_mask(detection, shape)[ey0:ey1, ex0:ex1] > 0
    for index in range(1, count):
        component = labels == index
        outside = component & ~inner
        # A detector foreground mask is a hypothesis, not proof that every
        # pixel belongs to the watermark.  A dark/colored component with clean
        # support across the boundary can legitimately overlap that mask (for
        # example a bubble arc underneath a translucent panel), so retain the
        # observed component and let identity/color evidence decide isolated
        # foreground pixels.
        inside = component & inner
        outside_count = int(np.count_nonzero(outside))
        inside_count = int(np.count_nonzero(inside))
        touches_inner_boundary = bool(
            np.any(component[inner])
            and (
                np.any(component[max(0, y0 - ey0):max(0, y0 - ey0 + 1), :])
                or np.any(component[min(component.shape[0], y1 - ey0 - 1):min(component.shape[0], y1 - ey0), :])
                or np.any(component[:, max(0, x0 - ex0):max(0, x0 - ex0 + 1)])
                or np.any(component[:, min(component.shape[1], x1 - ex0 - 1):min(component.shape[1], x1 - ex0)])
            )
        )
        if (outside_count >= 3 and inside_count >= 3) or (
            outside_count == 0
            and inside_count >= 12
            and touches_inner_boundary
        ):
            trusted_local |= inside

    trusted = _empty_mask(shape)
    trusted[ey0:ey1, ex0:ex1][trusted_local] = 255
    structure_pixels = int(np.count_nonzero(trusted))
    box_pixels = max(1, (x1 - x0) * (y1 - y0))
    structure_fraction = structure_pixels / box_pixels

    # Geometry support is computed from observed ink outside the box.  It is
    # intentionally strict: ambiguous candidates are reported, never drawn.
    crop_gray = gray
    edges = cv2.Canny(crop_gray, 45, 130)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(18, round(0.08 * (x1 - x0))),
        minLineLength=max(18, round(0.18 * (x1 - x0))),
        maxLineGap=6,
    )
    line_candidates: list[tuple[float, float, float]] = []
    if lines is not None:
        for xa, ya, xb, yb in lines[:, 0]:
            dx = float(xb - xa)
            dy = float(yb - ya)
            length = float(np.hypot(dx, dy))
            if length <= 0:
                continue
            # A supported corridor must cross the overlay footprint and have
            # meaningful context outside it.  The endpoint test is invariant
            # to page position and avoids the old location shortcut.
            gx_a, gy_a = xa + ex0, ya + ey0
            gx_b, gy_b = xb + ex0, yb + ey0
            in_a = x0 <= gx_a < x1 and y0 <= gy_a < y1
            in_b = x0 <= gx_b < x1 and y0 <= gy_b < y1
            if in_a == in_b:
                continue
            angle = abs(float(np.degrees(np.arctan2(dy, dx)))) % 180.0
            if min(10.0, abs(angle - 0.0), abs(angle - 180.0), abs(angle - 90.0)) > 10.0:
                continue
            line_candidates.append((length, angle, 1.0))
    horizontal = [line for line in line_candidates if min(line[1], 180.0 - line[1]) <= 12.0]
    vertical = [line for line in line_candidates if abs(line[1] - 90.0) <= 12.0]
    card_supported = (
        len(horizontal) >= 2
        or len(vertical) >= 2
        or (len(line_candidates) >= 2 and sum(line[0] for line in line_candidates) >= 0.55 * (x1 - x0))
    )

    circles = cv2.HoughCircles(
        cv2.medianBlur(crop_gray, 5),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(35, round(0.20 * (x1 - x0))),
        param1=100,
        param2=max(28, round(0.16 * (x1 - x0))),
        minRadius=max(24, round(0.18 * (x1 - x0))),
        maxRadius=max(45, round(1.35 * (x1 - x0))),
    )
    circle_scores: list[float] = []
    if circles is not None:
        for cx, cy, radius in np.round(circles[0]).astype(int):
            ring = np.zeros(crop_gray.shape, dtype=np.uint8)
            cv2.circle(ring, (int(cx), int(cy)), int(radius), 255, max(2, round(0.025 * radius)))
            ring_pixels = ring > 0
            outside = ring_pixels & ~inner
            support = float(np.mean(crop_gray[outside] < 145)) if np.any(outside) else 0.0
            outside_count = int(np.count_nonzero(outside & (crop_gray < 145)))
            if outside_count >= 20:
                circle_scores.append(support)
    circle_scores.sort(reverse=True)
    circle_supported = bool(circle_scores and circle_scores[0] >= 0.22)
    circle_ambiguous = len(circle_scores) > 1 and abs(circle_scores[0] - circle_scores[1]) < 0.06

    if circle_supported and not circle_ambiguous and not card_supported:
        geometry_kind = "circle"
        geometry_supported = True
        geometry_support = circle_scores[0]
        geometry_candidates = len(circle_scores)
    elif card_supported and not circle_supported:
        geometry_kind = "card"
        geometry_supported = True
        geometry_support = min(1.0, sum(line[0] for line in line_candidates) / max(1.0, 2.0 * (x1 - x0)))
        geometry_candidates = len(line_candidates)
    elif circle_scores or line_candidates:
        geometry_kind = "ambiguous"
        geometry_supported = False
        geometry_support = max(circle_scores or [0.0])
        geometry_candidates = len(circle_scores) + len(line_candidates)
    else:
        geometry_kind = None
        geometry_supported = False
        geometry_support = 0.0
        geometry_candidates = 0

    if geometry_kind == "circle":
        scene_class = "bubble-geometry"
    elif geometry_kind == "card":
        scene_class = "card-geometry"
    elif structure_pixels:
        scene_class = "structured-surface"
    elif surface.model == "affine":
        scene_class = "smooth-gradient"
    elif surface.model == "constant":
        scene_class = "flat-surface"
    else:
        mean_gray = float(np.mean(gray[inner])) if np.any(inner) else 0.0
        if mean_gray < 145:
            scene_class = "dark-artwork"
        elif surface.rejection_reason == "boundary-fit-not-validated":
            # A non-flat pale boundary is useful evidence for a coherent
            # legacy handoff, but it is not by itself enough to claim hidden
            # structure.  Keep it distinct from a detailed drawing.
            scene_class = "unknown"
        else:
            scene_class = "detailed-artwork"

    evidence: list[str] = []
    if structure_pixels:
        evidence.append("connected-structure-crosses-overlay-boundary")
    if surface.model:
        evidence.append(f"validated-{surface.model}-surface")
    if geometry_kind == "ambiguous":
        evidence.append("unsupported-or-ambiguous-geometry")
    confidence = min(
        1.0,
        0.35
        + (0.35 if surface.valid else 0.0)
        + min(0.25, structure_fraction * 2.0)
        + (0.20 if geometry_supported else 0.0),
    )
    scene = SceneEvidence(
        scene_class=scene_class,
        confidence=confidence,
        structure_pixels=structure_pixels,
        structure_fraction=structure_fraction,
        geometry_kind=geometry_kind,
        geometry_supported=geometry_supported,
        geometry_support=geometry_support,
        geometry_candidates=geometry_candidates,
        evidence=tuple(evidence),
    )
    return trusted, scene


def _identity_evidence(detection: Any) -> IdentityEvidence:
    name = str(getattr(detection, "name", "unknown"))
    if name == "baozimh_logo":
        family = "modern"
        default_variant = "modern-unresolved"
    elif name == "baozimh_top_left":
        family = "legacy"
        default_variant = "legacy-banner"
    elif name == "baozimh_promo":
        family = "promo"
        default_variant = "opaque-promo"
    else:
        family = "unknown"
        default_variant = "unknown"
    variant = str(
        getattr(detection, "variant", None)
        or getattr(detection, "identity_variant", None)
        or default_variant
    )
    confidence = float(np.clip(getattr(detection, "confidence", 0.0), 0.0, 1.0))
    variant_confidence = float(
        np.clip(
            getattr(detection, "variant_confidence", confidence if variant != "modern-unresolved" else confidence * 0.75),
            0.0,
            1.0,
        )
    )
    evidence = ("detector-confidence",)
    if variant == "modern-unresolved":
        evidence += ("variant-evidence-not-provided-by-detector",)
    else:
        evidence += ("variant-evidence-from-detector",)
    return IdentityEvidence(family, variant, confidence, variant_confidence, evidence)


def _registration_evidence(detection: Any) -> RegistrationEvidence:
    confidence = float(
        np.clip(
            getattr(
                detection,
                "registration_confidence",
                getattr(detection, "confidence", 0.0),
            ),
            0.0,
            1.0,
        )
    )
    offset = getattr(detection, "registration_offset", (0, 0))
    try:
        offset_tuple = (int(round(offset[0])), int(round(offset[1])))
    except (TypeError, IndexError, ValueError):
        offset_tuple = (0, 0)
    return RegistrationEvidence(
        confidence=confidence,
        offset=offset_tuple,
        source=str(getattr(detection, "registration_source", "detector-aligned")),
        evidence=("aligned-detection-box",),
    )


def _plan_route(
    identity: IdentityEvidence,
    scene: SceneEvidence,
    surface: SurfaceEvidence,
    missing_pixels: int,
    *,
    compatibility_mode: str = "auto",
) -> tuple[RoutingDecision, tuple[str, ...]]:
    qa: list[str] = []
    if identity.family == "promo":
        return RoutingDecision("promo-deterministic", "surface", "validated-opaque-promo"), tuple(qa)
    if identity.family not in {"modern", "legacy"}:
        qa.append("ambiguous_identity")
        return RoutingDecision("qa-hold", "none", "unsupported-overlay-family"), tuple(qa)

    if compatibility_mode != "auto" and identity.family == "legacy":
        route = f"legacy-{compatibility_mode}"
        if scene.geometry_kind == "ambiguous":
            qa.append("unsupported_geometry_reconstruction")
        return RoutingDecision(route, "localized" if compatibility_mode == "precise" else "broad-compatible", "explicit-compatibility-mode", compatibility_mode), tuple(qa)

    # A validated surface may still contain a small observed stroke.  Keep the
    # deterministic route for that case, but only after reducing its support by
    # the trusted structure mask.  Larger connected scenes require localized
    # handling so a numerically good pale fit cannot erase artwork.
    if surface.valid and (
        (identity.family == "legacy" and scene.structure_fraction < 0.10)
        or (identity.family == "modern" and scene.structure_fraction < 0.001)
    ):
        route = "surface-affine" if surface.model == "affine" else "surface-constant"
        return RoutingDecision(route, "deterministic", "validated-boundary-surface"), tuple(qa)

    if scene.geometry_kind == "ambiguous":
        qa.append("unsupported_geometry_reconstruction")
    if scene.geometry_kind == "circle" and not scene.geometry_supported:
        qa.append("unsupported_geometry_reconstruction")
    if identity.family == "legacy":
        if scene.scene_class in {"bubble-geometry", "card-geometry"}:
            qa.append("unsupported_geometry_reconstruction")
            return RoutingDecision("legacy-auto", "localized", "supported-geometry-fit-first"), tuple(qa)
        if scene.scene_class == "structured-surface":
            # A structured panel must first go through the empirical legacy
            # fit.  Strong fits on detailed pages (26/0018 and 26/0035) retain
            # the accepted auto route; the cleaner can downgrade a weak fit to
            # a precise/localized mask after seeing the actual fit evidence.
            if scene.structure_fraction < 0.10:
                return RoutingDecision("legacy-coherent", "broad-compatible", "connected-structure-without-supported-geometry"), tuple(qa)
            qa.append("structural_discontinuity")
            return RoutingDecision("legacy-auto", "localized", "structured-legacy-scene-fit-first"), tuple(qa)
        if scene.scene_class in {"dark-artwork", "detailed-artwork"}:
            qa.append("weak_overlay_fit")
            return RoutingDecision("legacy-auto", "localized", "legacy-source-fit-required"), tuple(qa)
        # A non-flat boundary is evidence for coherent reconstruction, not a
        # location rule.  It remains compatible with the established mode.
        return RoutingDecision("legacy-coherent", "broad-compatible", "surface-fit-rejected"), tuple(qa)

    if scene.scene_class == "dark-artwork":
        return RoutingDecision("modern-localized", "localized", "dark-scene-missing-information"), tuple(qa)
    if scene.scene_class in {"bubble-geometry", "card-geometry", "structured-surface"}:
        return RoutingDecision("modern-localized", "localized", "structured-scene-source-preserving"), tuple(qa)
    return RoutingDecision("modern-localized", "localized", "surface-fit-rejected"), tuple(qa)


def build_overlay_plan(
    image: Image.Image | np.ndarray,
    detection: Any,
    *,
    detection_index: int = 0,
    legacy_banner_mode: str = "auto",
) -> tuple[OverlayPlan, np.ndarray]:
    """Build one independent plan and its surface prediction image."""
    rgb = _as_rgb(image)
    shape = rgb.shape[:2]
    identity = _identity_evidence(detection)
    registration = _registration_evidence(detection)
    surface, prediction = _fit_surface_evidence(rgb, tuple(detection.box), identity.family)
    trusted, scene = _structure_and_geometry(rgb, detection, surface, prediction)
    template = _registered_template_mask(detection, shape)
    clipped = _box_slices(tuple(detection.box), shape)
    support = _empty_mask(shape)
    if clipped is not None:
        x0, y0, x1, y1 = clipped
        if identity.family == "legacy":
            # The translucent panel is a real support region even when the
            # foreground mask is sparse; the deterministic surface route is
            # allowed to restore its validated background.
            support[y0:y1, x0:x1] = 255
        elif identity.family == "promo":
            support = template
        else:
            support = template.copy()
            # A small registration-tolerant halo catches anti-aliased fringe
            # without turning the detector's foreground mask into a rectangle.
            if np.any(support):
                local = support[y0:y1, x0:x1]
                local = cv2.dilate(local, np.ones((3, 3), np.uint8), iterations=1)
                support[y0:y1, x0:x1] = local
    # Connected artwork is trustworthy only outside the registered overlay
    # foreground.  Treating the observed watermark glyphs themselves as
    # connected structure would restore the logo after every model pass.
    # The small modern registration halo can still protect a crossing stroke,
    # while the precise template support remains explicitly missing.
    trusted &= support
    crossing = np.zeros(shape, dtype=bool)
    crossing_input = (trusted > 0).astype(np.uint8)
    component_count, component_labels = cv2.connectedComponents(crossing_input, 8)
    if clipped is not None:
        x0, y0, x1, y1 = clipped
        side = max(2, round(min(x1 - x0, y1 - y0) * 0.03))
        for component_id in range(1, component_count):
            component = component_labels == component_id
            left = np.any(component[y0:y1, x0:min(x1, x0 + side)])
            right = np.any(component[y0:y1, max(x0, x1 - side):x1])
            top = np.any(component[y0:min(y1, y0 + side), x0:x1])
            bottom = np.any(component[max(y0, y1 - side):y1, x0:x1])
            ys, xs = np.where(component[y0:y1, x0:x1])
            component_w = int(xs.max() - xs.min() + 1) if xs.size else 0
            component_h = int(ys.max() - ys.min() + 1) if ys.size else 0
            horizontal_stroke = (
                left
                and right
                and component_h <= max(6, round(0.12 * (y1 - y0)))
            )
            vertical_stroke = (
                top
                and bottom
                and component_w <= max(6, round(0.12 * (x1 - x0)))
            )
            if horizontal_stroke or vertical_stroke:
                crossing |= component
    # Pixels coincident with registered foreground are mixed with the overlay
    # regardless of their resulting brightness.  Promote only independently
    # supported straight strokes which cross both sides of the overlay; all
    # other registered foreground remains missing/reconstructable.
    trusted[support > 0] = 0
    trusted[(support > 0) & crossing] = 255
    if identity.family == "modern" and clipped is not None:
        icon_box = _observed_logo_icon_box(rgb, tuple(detection.box))
        if icon_box is not None:
            ix0, iy0, ix1, iy1 = icon_box
            bx0, by0, _bx1, _by1 = clipped
            trusted[
                by0 + iy0:by0 + iy1,
                bx0 + ix0:bx0 + ix1,
            ] = 0
    missing = support.copy()
    missing[trusted > 0] = 0
    missing_pixels = int(np.count_nonzero(missing))
    routing, qa = _plan_route(
        identity,
        scene,
        surface,
        missing_pixels,
        compatibility_mode=legacy_banner_mode,
    )
    qa_list = list(qa)
    detector_evidence = getattr(detection, "evidence", None)
    if identity.family == "modern" and identity.variant in {
        "ambiguous",
        "modern-unresolved",
    }:
        qa_list.append("ambiguous_identity")
    if registration.confidence < 0.85:
        qa_list.append("uncertain_registration")
    for reason in getattr(detector_evidence, "reasons", ()):
        if reason in {"ambiguous_identity", "uncertain_registration"}:
            qa_list.append(reason)
    qa = tuple(dict.fromkeys(qa_list))
    if identity.family == "modern" and "variant_anchor_conflict" in getattr(detector_evidence, "reasons", ()):
        routing = RoutingDecision("qa-hold", "none", "variant-anchor-registration-conflict")
    deterministic = _empty_mask(shape)
    if routing.route.startswith("surface-") or routing.route == "promo-deterministic":
        deterministic = missing.copy()
    localized = missing.copy()
    if routing.generation_scope == "none":
        localized[:] = 0
    if routing.generation_scope == "deterministic":
        localized[:] = 0
    recovery = RecoveryEvidence(
        confidence=(
            min(1.0, 0.55 + surface.complexity_gain)
            if surface.valid
            else 0.0
        ),
        recoverable_pixels=0,
        deterministic_pixels=int(np.count_nonzero(deterministic)),
        localized_pixels=int(np.count_nonzero(localized)),
        broad_pixels=0,
        fit_valid=surface.valid,
        fit_stability=(
            max(0.0, 1.0 - surface.holdout_error / 20.0)
            if np.isfinite(surface.holdout_error)
            else 0.0
        ),
        evidence=("source-preserving-missing-mask",),
    )
    x, y, w, h = (int(v) for v in detection.box)
    overlay_id = str(
        getattr(detection, "overlay_id", None)
        or f"{identity.family}:{detection_index}:{x}:{y}:{w}:{h}"
    )
    provenance = {
        "planner": "recovery-plan-v1",
        "detection_confidence": round(float(identity.confidence), 6),
        "surface_model": surface.model,
        "surface_holdout_error": round(float(surface.holdout_error), 6) if np.isfinite(surface.holdout_error) else None,
        "surface_seam_error": round(float(surface.seam_error), 6) if np.isfinite(surface.seam_error) else None,
        "mask_semantics": "support-minus-trusted-source",
    }
    plan = OverlayPlan(
        overlay_id=overlay_id,
        detection_index=detection_index,
        name=str(getattr(detection, "name", "unknown")),
        box=(x, y, w, h),
        identity=identity,
        registration=registration,
        scene=scene,
        recovery=recovery,
        routing=routing,
        surface=surface,
        trusted_source_mask=trusted,
        recovered_mask=_empty_mask(shape),
        deterministic_mask=deterministic,
        missing_information_mask=missing,
        generated_mask=_empty_mask(shape),
        failed_mask=_empty_mask(shape),
        qa_reasons=qa,
        provenance=provenance,
    )
    return plan, prediction


def build_overlay_plans(
    image: Image.Image | np.ndarray,
    detections: Iterable[Any],
    *,
    legacy_banner_mode: str = "auto",
) -> tuple[list[OverlayPlan], dict[str, np.ndarray]]:
    """Build independent plans; predictions are keyed by stable overlay id."""
    plans: list[OverlayPlan] = []
    predictions: dict[str, np.ndarray] = {}
    for index, detection in enumerate(detections):
        plan, prediction = build_overlay_plan(
            image,
            detection,
            detection_index=index,
            legacy_banner_mode=legacy_banner_mode,
        )
        plans.append(plan)
        predictions[plan.overlay_id] = prediction
    return plans, predictions


def _residual_candidate_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    local_gray = cv2.medianBlur(gray, 31)
    local_sat = cv2.medianBlur(hsv[:, :, 1], 31)
    neutral = (hsv[:, :, 1] <= 30) & (gray >= 95) & (gray <= 245)
    desaturated = hsv[:, :, 1].astype(np.int16) <= local_sat.astype(np.int16) - 4
    # The known modern wordmark is a light/desaturated overlay.  Looking for
    # lightening rather than absolute contrast rejects dark comic ink and the
    # tiny isolated specks seen on 5/0030 while still catching a pale residual
    # over dark artwork.
    contrast = gray.astype(np.int16) - local_gray.astype(np.int16) >= 8
    return (neutral | desaturated) & contrast


def _modern_overlay_palette_mask(rgb: np.ndarray) -> np.ndarray:
    """Return pixels compatible with a flattened modern-logo rendering.

    This is evidence only, never a cleanup mask.  The modern marks contain a
    yellow icon and a pale, desaturated wordmark.  Keeping the palette test
    separate from the residual candidate map lets correction require support
    in both the output and the untouched source page.
    """

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    yellow = (
        (hsv[:, :, 0] >= 7)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 2] >= 110)
    )
    pale = (hsv[:, :, 1] <= 75) & (gray >= 145) & (gray <= 248)
    return yellow | pale


def evaluate_registered_residual(
    image: Image.Image | np.ndarray,
    detections: Iterable[Any],
    *,
    source_image: Image.Image | np.ndarray | None = None,
    authorized_mask: np.ndarray | None = None,
    artwork_veto_mask: np.ndarray | None = None,
    min_coverage: float = 0.08,
    min_area: int = 80,
    min_support: int = 20,
    min_mean_contrast: float = 20.0,
) -> ResidualCheck:
    """Check residual registered logo components using independent hard gates.

    A speck or isolated highlight is not enough to trigger another model pass.
    A component must simultaneously cover at least 8% of its registered
    support, contain at least 80 support pixels, have at least 20 candidate
    residual pixels, and exceed the local background by 20 gray levels on
    average.  The strength gate separates readable residual glyphs from weak
    Telea texture which happens to align with a large registered component.

    ``source_image``/``authorized_mask`` are optional so compatibility callers
    and synthetic tests retain the original API.  The runtime cleaner supplies
    both: automatic correction is then restricted to prior authorized support
    and independent source-palette evidence.  ``artwork_veto_mask`` protects
    trusted source structure when one is available.
    """
    rgb = _as_rgb(image)
    shape = rgb.shape[:2]
    candidate = _residual_candidate_mask(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    local_gray = cv2.medianBlur(gray, 31)
    contrast_delta = gray.astype(np.int16) - local_gray.astype(np.int16)
    source_rgb = _as_rgb(source_image) if source_image is not None else None
    if source_rgb is not None and source_rgb.shape != rgb.shape:
        source_rgb = None
    source_palette = (
        _modern_overlay_palette_mask(source_rgb)
        if source_rgb is not None
        else None
    )
    authorized = None
    if isinstance(authorized_mask, np.ndarray) and authorized_mask.shape == shape:
        authorized = authorized_mask > 0
    artwork_veto = None
    if isinstance(artwork_veto_mask, np.ndarray) and artwork_veto_mask.shape == shape:
        artwork_veto = artwork_veto_mask > 0
    operation = _empty_mask(shape)
    components: list[dict[str, Any]] = []
    qualifying = 0
    max_coverage = 0.0
    for detection in detections:
        if str(getattr(detection, "name", "")) != "baozimh_logo":
            continue
        template = _registered_template_mask(detection, shape)
        x, y, w, h = (int(v) for v in detection.box)
        clipped = _box_slices((x, y, w, h), shape)
        if clipped is None:
            continue
        x0, y0, x1, y1 = clipped
        local_template = template[y0:y1, x0:x1] > 0
        local_candidate = candidate[y0:y1, x0:x1]
        local_contrast = contrast_delta[y0:y1, x0:x1]
        local_output_hsv = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
        local_source_palette = (
            source_palette[y0:y1, x0:x1]
            if source_palette is not None
            else None
        )
        local_authorized = (
            authorized[y0:y1, x0:x1] if authorized is not None else None
        )
        local_artwork_veto = (
            artwork_veto[y0:y1, x0:x1] if artwork_veto is not None else None
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(local_template.astype(np.uint8), 8)
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            component = labels == index
            support = int(np.count_nonzero(local_candidate & component))
            coverage = support / area
            mean_contrast = (
                float(np.mean(local_contrast[local_candidate & component]))
                if support
                else 0.0
            )
            max_coverage = max(max_coverage, coverage)
            passes = (
                area >= min_area
                and support >= min_support
                and coverage >= min_coverage
                and mean_contrast >= min_mean_contrast
            )
            hits = local_candidate & component
            output_neutral_fraction = (
                float(np.mean(local_output_hsv[:, :, 1][hits] <= 45))
                if support
                else 0.0
            )
            source_palette_fraction = (
                float(np.mean(local_source_palette[hits]))
                if source_palette is not None and support
                else 0.0
            )
            source_copy_fraction = 0.0
            if source_rgb is not None and support:
                source_local = source_rgb[y0:y1, x0:x1]
                output_local = rgb[y0:y1, x0:x1]
                source_copy_fraction = float(
                    np.mean(
                        np.max(
                            np.abs(
                                source_local.astype(np.int16)
                                - output_local.astype(np.int16)
                            ),
                            axis=2,
                        )[hits]
                        <= 12
                    )
                )

            # The ordinary residual gate remains a QA signal.  Automatic
            # correction is stricter: a known registered component needs
            # independent source-palette evidence and neutral output pixels.
            # Strongly mixed/saturated output is a common signature of real
            # artwork beneath the logo (for example 9/52) and must not receive
            # a corrective mask.
            known_registered_variant = bool(
                str(getattr(detection, "variant", "")) in {"current", "older"}
                and float(
                    np.clip(
                        getattr(
                            detection,
                            "registration_confidence",
                            getattr(detection, "confidence", 0.0),
                        ),
                        0.0,
                        1.0,
                    )
                )
                >= 0.95
            )
            correction_strength = bool(
                area >= min_area
                and support >= max(min_support, 80)
                and coverage >= 0.07
                and mean_contrast >= min_mean_contrast
            )
            correction_eligible = bool(
                correction_strength
                and output_neutral_fraction >= 0.85
                and (
                    source_image is None
                    or source_palette_fraction >= 0.55
                )
                and (source_image is None or known_registered_variant)
                and (authorized is None or bool(np.any(local_authorized & hits)))
                and not (
                    local_artwork_veto is not None
                    and bool(np.any(local_artwork_veto & hits))
                )
            )
            correction_mode = None
            if correction_eligible:
                # When most of a small registered glyph/component is visibly
                # residual, the antialiased fringe between candidate pixels is
                # part of the same watermark.  Removing that whole *small*
                # component is still localized; broad low-coverage components
                # (the common artwork-overlap case) retain the candidate-only
                # operation below.
                correction_mode = (
                    "registered-component"
                    if coverage >= 0.30
                    and area <= max(2400, int(0.10 * (w * h)))
                    else "candidate-fringe"
                )
            components.append(
                {
                    "area": area,
                    "candidate_hits": support,
                    "coverage": round(float(coverage), 6),
                    "mean_contrast": round(mean_contrast, 6),
                    "qualifies": passes,
                    "output_neutral_fraction": round(output_neutral_fraction, 6),
                    "source_palette_fraction": round(source_palette_fraction, 6),
                    "source_copy_fraction": round(source_copy_fraction, 6),
                    "correction_eligible": correction_eligible,
                    "correction_mode": correction_mode,
                }
            )
            if passes:
                qualifying += 1
            if correction_eligible:
                # Keep the operation mask local to observed residual evidence;
                # a whole registered component is too broad for a retry.
                # Intersect after dilation so no new support is invented by
                # this post-clean pass.
                if correction_mode == "registered-component":
                    local_operation = component.copy()
                else:
                    local_operation = cv2.dilate(
                        hits.astype(np.uint8),
                        np.ones((3, 3), np.uint8),
                        iterations=1,
                    ) > 0
                if local_authorized is not None:
                    local_operation &= local_authorized
                if local_artwork_veto is not None:
                    local_operation &= ~local_artwork_veto
                operation[y0:y1, x0:x1][local_operation] = 255

    reasons = ("residual_watermark_evidence",) if qualifying else ()
    return ResidualCheck(
        candidate_mask=(candidate.astype(np.uint8) * 255),
        operation_mask=operation,
        components=tuple(components),
        should_flag=qualifying > 0,
        reasons=reasons,
    )


# Short aliases make the planner convenient for callers and tests while the
# descriptive names remain the canonical API.
plan_overlay = build_overlay_plan
plan_overlays = build_overlay_plans
check_registered_residual = evaluate_registered_residual
