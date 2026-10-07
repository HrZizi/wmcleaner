from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator, Mapping

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DetectionEvidence(Mapping[str, Any]):
    """Independent evidence carried by one detector result.

    ``Detection`` deliberately keeps its original four required constructor
    arguments.  Evidence is optional at the API boundary so callers that make
    synthetic detections (and older integrations) remain source compatible.
    Runtime detections populate the dimensions independently instead of
    collapsing them into one opaque quality score.  The mapping methods make
    the object convenient for debug/JSON callers while attribute access keeps
    the typed form pleasant for Python callers.
    """

    family: str
    variant: str = "ambiguous"
    mask_variant: str = "common"
    edge: str | None = None
    scale: float = 1.0
    registration_offset: tuple[int, int] = (0, 0)
    registration_uncertainty: tuple[int, int] = (0, 0)
    anchor_score: float = 0.0
    peak_margin: float = 0.0
    yellow_iou: float | None = None
    yellow_recall: float | None = None
    dark_pattern_correlation: float | None = None
    glyph_mode: str = "strict"
    identity_confidence: float = 0.0
    registration_confidence: float = 0.0
    variant_support: float = 0.0
    competing_variant_support: float = 0.0
    variant_margin: float = 0.0
    variant_winner_gap: float = 0.0
    joint_support: float = 0.0
    registration_anchor_retention: float | None = None
    reasons: tuple[str, ...] = ()

    _ALIASES: ClassVar[dict[str, str]] = {
        "identity": "variant",
        "identity_variant": "variant",
        "registration": "registration_confidence",
        "offset": "registration_offset",
        "edge_evidence": "edge_evidence",
        "dark_pattern_corr": "dark_pattern_correlation",
        "yellow_shape_iou": "yellow_iou",
        "expected_yellow_recall": "yellow_recall",
    }

    def __getitem__(self, key: str) -> Any:
        field = self._ALIASES.get(key, key)
        if field == "edge_evidence":
            return self.edge
        try:
            return getattr(self, field)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __iter__(self) -> Iterator[str]:
        yield from (
            "family",
            "variant",
            "mask_variant",
            "edge",
            "scale",
            "registration_offset",
            "registration_uncertainty",
            "anchor_score",
            "peak_margin",
            "yellow_iou",
            "yellow_recall",
            "dark_pattern_correlation",
            "glyph_mode",
            "identity_confidence",
            "registration_confidence",
            "variant_support",
            "competing_variant_support",
            "variant_margin",
            "variant_winner_gap",
            "joint_support",
            "registration_anchor_retention",
            "reasons",
        )

    def __len__(self) -> int:
        return 22

    def as_dict(self) -> dict[str, Any]:
        return {key: self[key] for key in self}

    @property
    def identity(self) -> str:
        return self.variant

    @property
    def registration(self) -> float:
        return self.registration_confidence

    @property
    def edge_evidence(self) -> str | None:
        return self.edge


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h
    mask: np.ndarray  # uint8, full image size, 0/255
    evidence: DetectionEvidence | None = None

    @property
    def variant(self) -> str | None:
        return self.evidence.variant if self.evidence is not None else None

    @property
    def edge(self) -> str | None:
        return self.evidence.edge if self.evidence is not None else None

    @property
    def identity_variant(self) -> str | None:
        return self.variant

    @property
    def variant_confidence(self) -> float:
        if self.evidence is None:
            return 0.0
        if self.evidence.variant == "ambiguous":
            return 0.0
        return max(
            0.0,
            min(
                1.0,
                float(self.evidence.variant_margin),
                float(self.evidence.variant_winner_gap),
            ),
        )

    @property
    def registration_confidence(self) -> float:
        return self.evidence.registration_confidence if self.evidence is not None else 0.0

    @property
    def registration_offset(self) -> tuple[int, int]:
        return self.evidence.registration_offset if self.evidence is not None else (0, 0)

    @property
    def registration_source(self) -> str:
        return "detector-evidence" if self.evidence is not None else "detector-aligned"


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    anchor: str
    reference_width: int
    reference_height: int | None
    box: tuple[int, int, int, int]
    min_confidence: float
    protect_dark: bool
    search_radius_px: int


# Legacy corner watermark variant. The current Baozimh logo watermark is
# position-independent along the top/bottom edge and is handled separately
# below. Coordinates are measured at a reference width of 850 px.
LEGACY_SPECS: tuple[TemplateSpec, ...] = (
    TemplateSpec(
        name="baozimh_top_left",
        anchor="top-left",
        reference_width=850,
        reference_height=None,
        box=(20, 20, 325, 92),
        min_confidence=0.60,
        protect_dark=True,
        search_radius_px=22,
    ),
)


# Current Baozimh watermark: yellow dumpling icon followed by pale gray
# ``包子漫畫`` text. Unlike the older watermark, the site places this mark at
# different horizontal positions but keeps it close to either the top or bottom
# page edge. The icon is therefore searched across the full edge strip.
LOGO_REFERENCE_WIDTH = 850
LOGO_BOX_SIZE = (320, 110)
LOGO_EDGE_OFFSET_PX = 34
LOGO_EDGE_TOLERANCE_PX = 18
# Corpus validation found that the only edge-icon matches below 0.90 were clean
# pages whose hands/line art happened to resemble the yellow dumpling glyph.
# Genuine modern marks, including the darker chapter-3 case, remain above 0.90.
LOGO_MIN_CONFIDENCE = 0.90

# A candidate that fails the strict glyph shape check can still be accepted
# when the independent visual evidence says that the icon is the known logo
# seen through an occluder.  These are intentionally stricter than the normal
# anchor threshold so a relaxed caller cannot turn this into a detector-wide
# threshold lowering.
LOGO_OCCLUSION_ANCHOR_GATE = 0.93
LOGO_OCCLUSION_PEAK_MARGIN_GATE = 0.01
LOGO_OCCLUSION_YELLOW_IOU_GATE = 0.88
LOGO_OCCLUSION_YELLOW_RECALL_GATE = 0.90
LOGO_OCCLUSION_DARK_CORRELATION_GATE = 0.45

# Some current logos are registered correctly but have their icon glyph
# occluded by page artwork.  Their yellow shape remains strong enough to
# identify the anchor, while the strict dark-glyph and the normal
# occlusion-consistent gates are intentionally too strict.  This handoff is
# current-variant-specific and keeps the generic modern threshold unchanged.
LOGO_CURRENT_OCCLUSION_YELLOW_IOU_GATE = 0.84
LOGO_CURRENT_OCCLUSION_YELLOW_RECALL_GATE = 0.89
LOGO_CURRENT_OCCLUSION_DARK_CORRELATION_GATE = 0.30
LOGO_CURRENT_OCCLUSION_VARIANT_SUPPORT_GATE = 0.55
LOGO_CURRENT_OCCLUSION_JOINT_SUPPORT_GATE = 0.55

# A margin alone can be produced by a tiny isolated icon fragment on an
# otherwise blank/pale page.  Require meaningful absolute support before
# exposing a variant-specific mask; weak support remains conservative common
# support even when the local control happens to be slightly worse.
LOGO_VARIANT_SUPPORT_GATE = 0.50

# A local peak margin avoids accepting a broad, repeated yellow texture as a
# registered banner.  The score and hard anchor gate above remain authoritative
# and this margin only disambiguates the location of a candidate.
LEGACY_BANNER_PEAK_MARGIN_GATE = 0.01

# The translucent ``latest free comics`` banner uses two yellow pointing-hand
# anchors.  It was originally hard-coded to the top-left corner, but the site
# also emits the exact same banner at the top-right.  Matching both anchors as
# one wide signature is much safer than the old mostly-pale fixed-box score:
# speech bubbles can look pale, but they do not contain the paired hands.
LEGACY_BANNER_EDGE_BAND_PX = 180
LEGACY_BANNER_MIN_CONFIDENCE = 0.90
LEGACY_BANNER_ANCHOR_GATE = 0.86

# Some chapter boundary pages contain an opaque Baozimh attribution panel: a
# centred dumpling icon/title plus two black text rows on a white strip.  This
# is a different object from the translucent 320x110 edge logo and needs the
# whole strip removed rather than a small logo-shaped mask.
PROMO_EDGE_BAND_PX = 240
PROMO_ANCHOR_MIN_CONFIDENCE = 0.86
PROMO_PANEL_ABOVE_ICON_PX = 25
PROMO_PANEL_BELOW_ICON_PX = 180


def _load_gray_asset(name: str) -> np.ndarray:
    asset = resources.files("wmcleaner").joinpath("templates", name)
    with resources.as_file(asset) as p:
        return np.array(Image.open(p).convert("L"), dtype=np.uint8)


def _load_rgb_asset(name: str) -> np.ndarray:
    asset = resources.files("wmcleaner").joinpath("templates", name)
    with resources.as_file(asset) as p:
        return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)


def _yellow_likelihood(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    # Broad enough for the yellow/orange hand/logo after anti-aliasing.
    result = (
        (h >= 7)
        & (h <= 42)
        & (s >= 55)
        & (v >= 125)
    ).astype(np.uint8) * 255
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return result


def _resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _logo_dark_glyph_valid(icon_gray: np.ndarray) -> bool:
    """Validate that a yellow anchor also contains the Baozimh inner glyph.

    Matching only the rounded yellow square can mistake warm skin/artwork at a
    page edge for the logo.  The real icon has a stable dark dumpling face well
    inside the yellow square; edge-art false positives instead concentrate their
    dark pixels along an outer contour.
    """
    if icon_gray.ndim != 2 or icon_gray.size == 0:
        return False

    dark_pixels = icon_gray < 120
    dark_fraction = float(np.mean(dark_pixels))
    if dark_fraction < 0.05 or dark_fraction > 0.16:
        return False

    ah, aw = icon_gray.shape
    inset_y = max(2, int(round(0.12 * ah)))
    inset_x = max(2, int(round(0.12 * aw)))
    if 2 * inset_y >= ah or 2 * inset_x >= aw:
        return False
    inner = dark_pixels[inset_y:ah - inset_y, inset_x:aw - inset_x]
    if inner.size == 0:
        return False

    inner_dark_fraction = float(np.mean(inner))
    border_count = int(np.count_nonzero(dark_pixels)) - int(np.count_nonzero(inner))
    border_area = dark_pixels.size - inner.size
    border_dark_fraction = float(border_count / border_area) if border_area else 0.0

    return (
        inner_dark_fraction >= 0.06
        and inner_dark_fraction >= 2.0 * border_dark_fraction
    )


def _template_peak_metrics(
    target: np.ndarray,
    template: np.ndarray,
    *,
    method: int = cv2.TM_CCORR_NORMED,
) -> tuple[float, float, tuple[int, int]]:
    """Return best score, isolated-peak margin, and best location.

    The second peak is measured outside a template-sized neighbourhood.  That
    keeps a broad plateau around one match from masquerading as independent
    registration evidence while retaining the existing OpenCV score semantics.
    """
    if target.ndim != 2 or template.ndim != 2:
        return 0.0, 0.0, (0, 0)
    if (
        target.shape[0] < template.shape[0]
        or target.shape[1] < template.shape[1]
    ):
        return 0.0, 0.0, (0, 0)

    result = cv2.matchTemplate(target, template, method)
    if result.size == 0:
        return 0.0, 0.0, (0, 0)
    _, max_value, _, max_loc = cv2.minMaxLoc(result)
    score = float(max_value) if np.isfinite(max_value) else 0.0

    # Suppress the same physical candidate before measuring the runner-up.
    # ``-inf`` is safe for all supported normalized match methods.
    second_result = result.copy()
    radius_x = max(1, int(round(template.shape[1] * 0.45)))
    radius_y = max(1, int(round(template.shape[0] * 0.45)))
    x0 = max(0, int(max_loc[0]) - radius_x)
    x1 = min(second_result.shape[1], int(max_loc[0]) + radius_x + 1)
    y0 = max(0, int(max_loc[1]) - radius_y)
    y1 = min(second_result.shape[0], int(max_loc[1]) + radius_y + 1)
    second_result[y0:y1, x0:x1] = -np.inf
    second = float(np.max(second_result)) if second_result.size else 0.0
    if not np.isfinite(second):
        second = 0.0
    margin = max(0.0, score - second)
    return score, margin, (int(max_loc[0]), int(max_loc[1]))


def _yellow_shape_metrics(
    icon_rgb: np.ndarray,
    anchor_template: np.ndarray,
) -> tuple[float, float]:
    """Measure observed-yellow IoU and expected-yellow recall independently."""
    if icon_rgb.ndim != 3 or icon_rgb.shape[2] != 3 or icon_rgb.size == 0:
        return 0.0, 0.0
    expected = _resize_mask(
        anchor_template,
        icon_rgb.shape[1],
        icon_rgb.shape[0],
    ) > 0
    observed = _yellow_likelihood(icon_rgb) > 0
    expected_count = int(np.count_nonzero(expected))
    union_count = int(np.count_nonzero(expected | observed))
    if expected_count == 0 or union_count == 0:
        return 0.0, 0.0
    intersection = int(np.count_nonzero(expected & observed))
    return float(intersection / union_count), float(intersection / expected_count)


def _logo_dark_pattern_correlation(
    icon_rgb: np.ndarray,
    anchor_template: np.ndarray,
) -> float:
    """Compare the matched icon's dark pattern with the registered reference."""
    if icon_rgb.ndim != 3 or icon_rgb.shape[2] != 3 or icon_rgb.size == 0:
        return 0.0
    reference = _load_rgb_asset("baozimh_bottom_right.png")
    full_anchor = _load_gray_asset("baozimh_bottom_right_anchor.png")
    ys, xs = np.where(full_anchor > 0)
    if len(xs) == 0:
        return 0.0
    ref_crop = reference[int(ys.min()):int(ys.max() + 1), int(xs.min()):int(xs.max() + 1)]
    if ref_crop.size == 0:
        return 0.0
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(icon_rgb, cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.resize(
        ref_gray,
        (target_gray.shape[1], target_gray.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # The strict glyph validator deliberately reasons about individual dark
    # pixels.  This secondary signal answers a different question: does the
    # low-frequency dark face/pattern still agree with the registered icon
    # when comic ink or lettering occludes parts of it?  A small blur on both
    # sides makes that comparison tolerant of a one-pixel registration drift
    # and high-frequency source artwork without changing the strict gate.
    ref_gray = cv2.GaussianBlur(ref_gray, (7, 7), 0)
    target_gray = cv2.GaussianBlur(target_gray, (7, 7), 0)

    # The yellow support is the registered region.  Ignoring pixels outside it
    # prevents page background texture from dominating this identity check.
    support = _resize_mask(
        anchor_template,
        target_gray.shape[1],
        target_gray.shape[0],
    ) > 0
    if int(np.count_nonzero(support)) < 8:
        return 0.0
    target_values = target_gray[support].astype(np.float32)
    reference_values = ref_gray[support].astype(np.float32)
    target_values -= float(np.mean(target_values))
    reference_values -= float(np.mean(reference_values))
    denominator = float(np.linalg.norm(target_values) * np.linalg.norm(reference_values))
    if denominator <= 1e-6:
        return 0.0
    correlation = float(np.dot(target_values, reference_values) / denominator)
    return max(-1.0, min(1.0, correlation))


def _box_for_image(spec: TemplateSpec, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    ih, iw = image_shape[:2]
    scale = iw / spec.reference_width
    x, y, w, h = spec.box
    sw = max(1, round(w * scale))
    sh = max(1, round(h * scale))

    if spec.anchor == "top-left":
        sx = round(x * scale)
        sy = round(y * scale)
    elif spec.anchor == "bottom-right":
        # Preserve offsets from the bottom/right. This works better for pages
        # whose height differs while keeping the same 850 px width.
        assert spec.reference_height is not None
        right = spec.reference_width - (x + w)
        bottom = spec.reference_height - (y + h)
        sx = iw - round((right + w) * scale)
        sy = ih - round((bottom + h) * scale)
    else:
        raise ValueError(f"Unsupported anchor: {spec.anchor}")

    sx = max(0, min(iw - sw, sx))
    sy = max(0, min(ih - sh, sy))
    return sx, sy, sw, sh


def _best_anchor_offset(
    image_rgb: np.ndarray,
    expected_box: tuple[int, int, int, int],
    anchor_template: np.ndarray,
    search_radius: int,
) -> tuple[float, int, int]:
    x, y, w, h = expected_box
    ih, iw = image_rgb.shape[:2]

    anchor_template = _resize_mask(anchor_template, w, h)
    # Shrink the template to the non-zero yellow content. Matching a mostly
    # black template is noisy, so we compare only the distinctive anchor area.
    ys, xs = np.where(anchor_template > 0)
    if len(xs) == 0:
        return 0.0, 0, 0
    ax0, ax1 = xs.min(), xs.max() + 1
    ay0, ay1 = ys.min(), ys.max() + 1
    anchor_crop = anchor_template[ay0:ay1, ax0:ax1]

    r = max(4, search_radius)
    sx0 = max(0, x + ax0 - r)
    sy0 = max(0, y + ay0 - r)
    sx1 = min(iw, x + ax1 + r)
    sy1 = min(ih, y + ay1 + r)
    if sx1 - sx0 < anchor_crop.shape[1] or sy1 - sy0 < anchor_crop.shape[0]:
        return 0.0, 0, 0

    target = _yellow_likelihood(image_rgb)[sy0:sy1, sx0:sx1]
    # Gaussian blur makes the score tolerant of anti-aliasing and slight size drift.
    target_f = cv2.GaussianBlur(target, (3, 3), 0).astype(np.float32) / 255.0
    templ_f = cv2.GaussianBlur(anchor_crop, (3, 3), 0).astype(np.float32) / 255.0

    result = cv2.matchTemplate(target_f, templ_f, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if not np.isfinite(max_val):
        max_val = 0.0

    found_x = sx0 + max_loc[0]
    found_y = sy0 + max_loc[1]
    expected_anchor_x = x + ax0
    expected_anchor_y = y + ay0
    return float(max_val), int(found_x - expected_anchor_x), int(found_y - expected_anchor_y)


def _foreground_presence_score(image_rgb: np.ndarray, box: tuple[int, int, int, int], mask: np.ndarray) -> float:
    x, y, w, h = box
    roi = image_rgb[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    m = _resize_mask(mask, w, h) > 0
    if not np.any(m):
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # Watermark foreground is mostly pale gray plus yellow/orange accents.
    yellow = (hsv[:, :, 0] >= 7) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 115)
    pale = (hsv[:, :, 1] <= 75) & (gray >= 145) & (gray <= 248)
    compatible = yellow | pale
    return float(np.mean(compatible[m]))


def _compatible_foreground_map(roi: np.ndarray) -> np.ndarray:
    """Return a palette-only foreground map for local variant comparisons."""
    if roi.ndim != 3 or roi.shape[2] != 3 or roi.size == 0:
        return np.zeros(roi.shape[:2], dtype=bool)
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    yellow = (
        (hsv[:, :, 0] >= 7)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] >= 45)
        & (hsv[:, :, 2] >= 115)
    )
    pale = (hsv[:, :, 1] <= 75) & (gray >= 145) & (gray <= 248)
    return yellow | pale


def _translate_local_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a local mask without changing its dimensions."""
    if mask.ndim != 2 or mask.size == 0 or (dx == 0 and dy == 0):
        return mask.copy()
    translated = cv2.warpAffine(
        mask,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return translated


def _mask_support_score(compatible: np.ndarray, mask: np.ndarray) -> float:
    support = mask > 0
    if not np.any(support):
        return 0.0
    return float(np.mean(compatible[support]))


def _variant_support_evidence(
    roi: np.ndarray,
    variant_mask: np.ndarray,
    competing_mask: np.ndarray,
    *,
    search_radius: int,
) -> tuple[float, float, float, tuple[int, int], np.ndarray]:
    """Score exclusive support against nearby translated controls.

    The candidate's exclusive pixels are searched over a small registration
    neighbourhood.  The control score is the median of translated copies away
    from the winning offset, so a broad pale page region cannot establish a
    variant identity merely by occupancy.
    """
    compatible = _compatible_foreground_map(roi)
    variant = _resize_mask(variant_mask, roi.shape[1], roi.shape[0]) > 0
    competing = _resize_mask(competing_mask, roi.shape[1], roi.shape[0]) > 0
    exclusive = (variant & ~competing).astype(np.uint8) * 255
    if int(np.count_nonzero(exclusive)) < 8:
        empty = np.zeros_like(exclusive)
        return 0.0, 0.0, 0.0, (0, 0), empty

    radius = max(1, int(search_radius))
    candidates: list[tuple[float, int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = _translate_local_mask(exclusive, dx, dy)
            score = _mask_support_score(compatible, shifted)
            candidates.append((score, dx, dy))

    best_score, best_dx, best_dy = max(candidates, key=lambda item: item[0])
    controls = [
        score
        for score, dx, dy in candidates
        if abs(dx - best_dx) + abs(dy - best_dy) >= max(2, radius // 2)
    ]
    control = float(np.median(controls)) if controls else 0.0
    margin = max(0.0, float(best_score - control))
    registered = _translate_local_mask(
        _resize_mask(variant_mask, roi.shape[1], roi.shape[0]),
        best_dx,
        best_dy,
    )
    return float(best_score), control, margin, (best_dx, best_dy), registered


def _select_logo_variant(
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    search_radius: int = 8,
) -> tuple[np.ndarray, str, str, float, float, float, float, tuple[int, int]]:
    """Select a registered logo variant or return conservative common support.

    Returns ``(mask, identity, mask_variant, support, competing, margin,
    winner_gap, offset)``.  The two explicit margins are intentionally kept
    separate: one establishes that the winning variant is locally supported,
    the other establishes separation from its competitor.
    """
    x, y, w, h = box
    roi = image_rgb[y:y + h, x:x + w]
    current = _resize_mask(_load_gray_asset("baozimh_logo_mask.png"), w, h)
    older = _resize_mask(_load_gray_asset("baozimh_bottom_right_mask.png"), w, h)
    if roi.size == 0:
        return current, "ambiguous", "common", 0.0, 0.0, 0.0, 0.0, (0, 0)

    current_evidence = _variant_support_evidence(
        roi,
        current,
        older,
        search_radius=search_radius,
    )
    older_evidence = _variant_support_evidence(
        roi,
        older,
        current,
        search_radius=search_radius,
    )
    candidates = (
        ("current", current_evidence),
        ("older", older_evidence),
    )
    winner_name, winner = max(candidates, key=lambda item: item[1][2])
    _runner_name, runner = min(candidates, key=lambda item: item[1][2])
    # The separation gate compares support-vs-control margins, not raw
    # occupancy.  Both variants can have similar palette support on a pale
    # page; only a locally distinctive margin should establish identity.
    winner_gap = max(0.0, float(winner[2] - runner[2]))
    support, _control, margin, offset, registered = winner

    if (
        support >= LOGO_VARIANT_SUPPORT_GATE
        and margin >= 0.05
        and winner_gap >= 0.04
    ):
        return (
            registered,
            winner_name,
            winner_name,
            float(support),
            float(runner[0]),
            float(margin),
            float(winner_gap),
            offset,
        )

    # Ambiguous identity authorizes common support plus pixels in either
    # registered variant which are visibly compatible with the overlay
    # palette.  This is observed support, not a union-shaped cleanup mask: dark
    # lettering/artwork and absent variant-exclusive pixels stay excluded.
    current_registered = current_evidence[4]
    older_registered = older_evidence[4]
    common = cv2.bitwise_and(current_registered, older_registered)
    union = cv2.bitwise_or(current_registered, older_registered)
    observed = union.copy()
    observed[~_compatible_foreground_map(roi)] = 0
    observed = cv2.bitwise_or(common, observed)
    return (
        observed,
        "ambiguous",
        "common",
        float(support),
        float(runner[0]),
        float(margin),
        float(winner_gap),
        offset,
    )


def _logo_variant_joint_support(
    roi: np.ndarray,
    registered_mask: np.ndarray,
    anchor_template: np.ndarray,
    *,
    yellow_recall: float,
) -> float:
    """Measure independent support for the icon and current wordmark.

    Variant-exclusive occupancy is useful for choosing between the current
    and older masks, but it is not by itself enough to recover a heavily
    occluded icon: a pale speech bubble can occupy much of either mask.  The
    current-logo fallback therefore requires visible support in the registered
    wordmark *and* in the independently matched yellow icon.  The lower of the
    two scores is returned so one strong component cannot hide a missing one.
    """
    if roi.ndim != 3 or roi.shape[2] != 3 or roi.size == 0:
        return 0.0

    mask = _resize_mask(registered_mask, roi.shape[1], roi.shape[0]) > 0
    anchor = _resize_mask(anchor_template, roi.shape[1], roi.shape[0]) > 0
    wordmark = mask & ~anchor
    wordmark_pixels = int(np.count_nonzero(wordmark))
    if wordmark_pixels < 20:
        return 0.0

    compatible = _compatible_foreground_map(roi)
    wordmark_support = float(np.mean(compatible[wordmark]))
    return max(0.0, min(1.0, wordmark_support, float(yellow_recall)))


def _build_aligned_mask(
    image_rgb: np.ndarray,
    spec: TemplateSpec,
    box: tuple[int, int, int, int],
    precise_template: np.ndarray,
    aggressive: bool,
    dilation: int,
) -> np.ndarray:
    ih, iw = image_rgb.shape[:2]
    x, y, w, h = box
    out = np.zeros((ih, iw), dtype=np.uint8)

    roi = image_rgb[y:y+h, x:x+w]
    protected_dark = None

    if aggressive:
        local = np.full((h, w), 255, dtype=np.uint8)
        if spec.protect_dark:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            protected_dark = (gray < 105).astype(np.uint8)
            protected_dark = cv2.dilate(protected_dark, np.ones((3, 3), np.uint8), iterations=1) > 0
            local[protected_dark] = 0
    else:
        local = _resize_mask(precise_template, w, h)

        # Only keep template pixels that still look compatible with the visible
        # watermark palette in the target page. This is important when the
        # watermark is spatially behind/occluded by a speech bubble: the template
        # may pass through lettering, but those pixels should not be erased.
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # On the speech-bubble-prone top-left variant, intersect the template
        # with colors that still look like the visible overlay. This prevents an
        # occluded watermark from causing legitimate bubble text to be masked.
        if spec.protect_dark:
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            yellow = (
                (hsv[:, :, 0] >= 7)
                & (hsv[:, :, 0] <= 42)
                & (hsv[:, :, 1] >= 45)
                & (hsv[:, :, 2] >= 115)
            )
            pale = (hsv[:, :, 1] <= 75) & (gray >= 145) & (gray <= 248)
            compatible = yellow | pale
            local[~compatible] = 0

            # Speech-bubble text and line art are usually very dark. Protect a
            # slightly expanded version of those pixels, and re-apply the protection
            # after mask dilation so dilation cannot grow back over the lettering.
            protected_dark = (gray < 105).astype(np.uint8)
            protected_dark = cv2.dilate(protected_dark, np.ones((3, 3), np.uint8), iterations=1) > 0
            local[protected_dark] = 0

    if dilation > 0:
        k = 2 * dilation + 1
        local = cv2.dilate(local, np.ones((k, k), np.uint8), iterations=1)

    if protected_dark is not None:
        local[protected_dark] = 0

    out[y:y+h, x:x+w] = np.maximum(out[y:y+h, x:x+w], local)
    return out


def _logo_variant_mask(
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    variant: str | None = None,
) -> np.ndarray:
    """Return a registered or conservative common-support logo mask.

    ``variant=None`` preserves the old helper's call shape while using the new
    local support evidence.  Callers that already made an identity decision can
    request ``current``/``older`` explicitly without repeating the occupancy
    heuristic that previously selected the old URL row.
    """
    current = _load_gray_asset("baozimh_logo_mask.png")
    older = _load_gray_asset("baozimh_bottom_right_mask.png")
    if variant == "current":
        return current
    if variant == "older":
        return older
    if variant in {"common", "ambiguous"}:
        return cv2.bitwise_and(current, older)
    selected, _identity, _mask_variant, *_ = _select_logo_variant(image_rgb, box)
    return selected


def _extend_logo_mask_palette_halo(
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    precise_template: np.ndarray,
    *,
    halo_px: int,
) -> np.ndarray:
    """Grow a modern-logo mask only onto pixels that still look like the overlay.

    Tight binary masks are good at preserving comic line art, but they leave the
    anti-aliased fringe of the semi-transparent yellow/gray Baozimh logo visible.
    LaMa can then copy that fringe back into the hole, producing the yellow box
    and pale ``包子漫畫`` ghosts seen in real chapter output.

    Instead of blindly dilating the whole mask, grow it a few pixels and keep only
    halo pixels whose target colors are compatible with the known watermark. This
    captures translucent edge pixels while avoiding nearby black panel borders and
    most unrelated artwork.
    """
    x, y, w, h = box
    roi = image_rgb[y:y + h, x:x + w]
    local = _resize_mask(precise_template, w, h)
    if roi.size == 0 or halo_px <= 0:
        return local

    k = 2 * halo_px + 1
    halo = cv2.dilate(local, np.ones((k, k), np.uint8), iterations=1) > 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    yellow = (
        (hsv[:, :, 0] >= 7)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 2] >= 110)
    )
    # The text is intentionally very faint. Keep the upper gray bound high so
    # compressed/anti-aliased edge pixels are included, but require low
    # saturation so colored artwork is not swallowed by the halo.
    pale = (hsv[:, :, 1] <= 55) & (gray >= 150) & (gray <= 253)
    compatible = yellow | pale

    expanded = local.copy()
    expanded[halo & compatible] = 255
    return expanded


def _observed_logo_icon_box(
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Locate the opaque yellow icon inside an aligned modern-logo box."""
    x, y, w, h = box
    roi = image_rgb[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    yellow = (_yellow_likelihood(roi) > 0).astype(np.uint8)
    yellow[:, max(1, int(round(0.42 * roi.shape[1]))):] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(yellow, 8)
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[index, cv2.CC_STAT_AREA]) < max(20, int(0.002 * w * h)):
        return None
    ix = int(stats[index, cv2.CC_STAT_LEFT])
    iy = int(stats[index, cv2.CC_STAT_TOP])
    iw = int(stats[index, cv2.CC_STAT_WIDTH])
    ih = int(stats[index, cv2.CC_STAT_HEIGHT])
    pad = max(2, int(round(w / 160)))
    return (
        max(0, ix - pad),
        max(0, iy - pad),
        min(roi.shape[1], ix + iw + pad),
        min(roi.shape[0], iy + ih + pad),
    )


def _detect_legacy_banners(
    rgb: np.ndarray,
    *,
    aggressive: bool,
    dilation: int,
    min_confidence: float | None,
) -> list[Detection]:
    """Detect the paired-hand legacy banner symmetrically at either edge."""
    ih, iw = rgb.shape[:2]
    spec = LEGACY_SPECS[0]
    scale = iw / spec.reference_width
    box_w = max(1, round(spec.box[2] * scale))
    box_h = max(1, round(spec.box[3] * scale))

    full_anchor = _resize_mask(
        _load_gray_asset(f"{spec.name}_anchor.png"), box_w, box_h
    )
    ys, xs = np.where(full_anchor > 0)
    if len(xs) == 0:
        return []
    ax0, ax1 = int(xs.min()), int(xs.max() + 1)
    ay0, ay1 = int(ys.min()), int(ys.max() + 1)
    anchor = full_anchor[ay0:ay1, ax0:ax1]

    anchor_f = anchor.astype(np.float32) / 255.0
    threshold = (
        min_confidence
        if min_confidence is not None
        else LEGACY_BANNER_MIN_CONFIDENCE
    )
    band_h = min(
        ih,
        max(anchor.shape[0], round(LEGACY_BANNER_EDGE_BAND_PX * scale)),
    )
    yellow = _yellow_likelihood(rgb).astype(np.float32) / 255.0

    results: list[Detection] = []
    windows = (
        ("top", 0, band_h),
        ("bottom", max(0, ih - band_h), ih),
    )
    for edge, sy0, sy1 in windows:
        target = yellow[sy0:sy1]
        if (
            target.shape[0] < anchor_f.shape[0]
            or target.shape[1] < anchor_f.shape[1]
        ):
            continue

        anchor_score, peak_margin, max_loc = _template_peak_metrics(
            target,
            anchor_f,
            method=cv2.TM_CCORR_NORMED,
        )
        # Keep a non-negotiable anchor gate even when a caller lowers the
        # generic confidence threshold.  The margin makes the location itself
        # evidence, rather than accepting a repeated yellow texture.
        if anchor_score < max(LEGACY_BANNER_ANCHOR_GATE, threshold):
            continue
        if peak_margin < LEGACY_BANNER_PEAK_MARGIN_GATE:
            continue

        anchor_x = int(max_loc[0])
        anchor_y = int(sy0 + max_loc[1])
        x = max(0, min(iw - box_w, anchor_x - ax0))
        y = max(0, min(ih - box_h, anchor_y - ay0))
        box = (int(x), int(y), int(box_w), int(box_h))
        precise = _load_gray_asset(f"{spec.name}_mask.png")
        mask = _build_aligned_mask(
            rgb, spec, box, precise, aggressive, dilation
        )
        if np.count_nonzero(mask) == 0:
            continue

        registration_confidence = max(
            0.0,
            min(1.0, 0.85 * anchor_score + 0.15 * min(1.0, peak_margin / 0.10)),
        )
        evidence = DetectionEvidence(
            family="legacy-banner",
            variant="legacy-banner",
            mask_variant="legacy-banner",
            edge=edge,
            scale=float(scale),
            registration_offset=(0, 0),
            registration_uncertainty=(
                max(1, round(2 * scale)),
                max(1, round(2 * scale)),
            ),
            anchor_score=anchor_score,
            peak_margin=peak_margin,
            glyph_mode="paired-hand-anchor",
            identity_confidence=anchor_score,
            registration_confidence=registration_confidence,
        )
        results.append(Detection(spec.name, anchor_score, box, mask, evidence))

    return results


def _detect_promo_blocks(rgb: np.ndarray) -> list[Detection]:
    """Detect opaque white Baozimh attribution strips at page boundaries."""
    ih, iw = rgb.shape[:2]
    scale = iw / LOGO_REFERENCE_WIDTH
    full_anchor = _load_gray_asset("baozimh_bottom_right_anchor.png")
    ys, xs = np.where(full_anchor > 0)
    if len(xs) == 0:
        return []
    anchor = full_anchor[int(ys.min()):int(ys.max() + 1), int(xs.min()):int(xs.max() + 1)]
    aw = max(1, round(anchor.shape[1] * scale))
    ah = max(1, round(anchor.shape[0] * scale))
    anchor_f = _resize_mask(anchor, aw, ah).astype(np.float32) / 255.0
    yellow = _yellow_likelihood(rgb).astype(np.float32) / 255.0
    band_h = min(ih, max(ah, round(PROMO_EDGE_BAND_PX * scale)))

    results: list[Detection] = []
    windows = (
        (0, band_h),
        (max(0, ih - band_h), ih),
    )
    for sy0, sy1 in windows:
        target = yellow[sy0:sy1]
        if target.shape[0] < ah or target.shape[1] < aw:
            continue
        matched = cv2.matchTemplate(target, anchor_f, cv2.TM_CCORR_NORMED)

        # A boundary page can contain a normal translucent logo and the opaque
        # attribution panel simultaneously.  Inspect several distinct anchor
        # candidates rather than retaining only the strongest one.
        for _ in range(4):
            _, max_val, _, max_loc = cv2.minMaxLoc(matched)
            score = float(max_val) if np.isfinite(max_val) else 0.0
            if score < PROMO_ANCHOR_MIN_CONFIDENCE:
                break
            anchor_x = int(max_loc[0])
            anchor_y = int(sy0 + max_loc[1])

            mx0 = max(0, max_loc[0] - aw)
            mx1 = min(matched.shape[1], max_loc[0] + aw)
            my0 = max(0, max_loc[1] - ah)
            my1 = min(matched.shape[0], max_loc[1] + ah)
            matched[my0:my1, mx0:mx1] = 0.0

            # The opaque promo icon is centred at roughly 37% of page width.
            # Position is a useful guard against ordinary edge logos on white
            # speech bubbles or gutters.
            if not (0.30 * iw <= anchor_x <= 0.45 * iw):
                continue
            icon = rgb[anchor_y:anchor_y + ah, anchor_x:anchor_x + aw]
            if icon.shape[:2] != (ah, aw):
                continue
            if not _logo_dark_glyph_valid(cv2.cvtColor(icon, cv2.COLOR_RGB2GRAY)):
                continue

            y0 = max(0, anchor_y - round(PROMO_PANEL_ABOVE_ICON_PX * scale))
            y1 = min(ih, anchor_y + round(PROMO_PANEL_BELOW_ICON_PX * scale))
            panel = rgb[y0:y1]
            if panel.size == 0:
                continue
            panel_hsv = cv2.cvtColor(panel, cv2.COLOR_RGB2HSV)
            white_fraction = float(np.mean(
                (panel_hsv[:, :, 1] < 40) & (panel_hsv[:, :, 2] > 215)
            ))

            tx0 = min(iw, anchor_x + aw)
            tx1 = min(iw, anchor_x + round(340 * scale))
            ty1 = min(ih, anchor_y + ah)
            title = rgb[anchor_y:ty1, tx0:tx1]
            title_dark = 0.0
            if title.size:
                title_dark = float(np.mean(
                    cv2.cvtColor(title, cv2.COLOR_RGB2GRAY) < 110
                ))
            if white_fraction < 0.78 or title_dark < 0.04:
                continue

            mask = np.zeros((ih, iw), dtype=np.uint8)
            mask[y0:y1, :] = 255
            confidence = min(
                1.0,
                0.70 * score
                + 0.15 * white_fraction
                + 0.15 * min(1.0, title_dark / 0.08),
            )
            evidence = DetectionEvidence(
                family="promo-block",
                variant="promo",
                mask_variant="promo-strip",
                edge="top" if sy0 == 0 else "bottom",
                scale=float(scale),
                anchor_score=score,
                identity_confidence=score,
                registration_confidence=score,
            )
            results.append(
                Detection(
                    "baozimh_promo",
                    confidence,
                    (0, y0, iw, y1 - y0),
                    mask,
                    evidence,
                )
            )

    # Very short pages can make the top/bottom search bands overlap.  Collapse
    # duplicate detections of the same strip.
    unique: list[Detection] = []
    for candidate in results:
        if any(candidate.box == existing.box for existing in unique):
            continue
        unique.append(candidate)
    return unique


def _logo_registration_anchor_retention(
    yellow: np.ndarray,
    anchor: np.ndarray,
    origin: tuple[int, int],
    offset: tuple[int, int],
    uncertainty: int,
) -> float:
    """Check palette registration against this instance's independent icon.

    Compare within raster-phase uncertainty, relative to the matched anchor,
    never relative to a fixed page position or a corpus-average offset.
    """
    h, w = anchor.shape
    anchor_norm = float(np.linalg.norm(anchor))

    def score(dx: int, dy: int) -> float:
        x, y = origin[0] + dx, origin[1] + dy
        if x < 0 or y < 0 or x + w > yellow.shape[1] or y + h > yellow.shape[0]:
            return 0.0
        patch = yellow[y:y + h, x:x + w]
        denominator = anchor_norm * float(np.linalg.norm(patch))
        return float(np.sum(anchor * patch) / denominator) if denominator else 0.0

    reference = score(0, 0)
    if reference <= 0:
        return 0.0
    dx, dy = offset
    best = max(
        score(dx + sx, dy + sy)
        for sy in range(-uncertainty, uncertainty + 1)
        for sx in range(-uncertainty, uncertainty + 1)
    )
    return min(1.0, best / reference)


def _detect_edge_logos(
    rgb: np.ndarray,
    *,
    aggressive: bool,
    dilation: int,
    min_confidence: float | None,
) -> list[Detection]:
    """Detect the modern Baozimh logo anywhere along the top/bottom edge.

    The old implementation assumed one fixed bottom-right location. Current
    Baozimh pages move the same logo horizontally (bottom-left, top-left,
    top-right, etc.). The yellow icon itself remains highly stable, including
    its size and its roughly 34 px edge offset at an 850 px page width.
    """
    ih, iw = rgb.shape[:2]
    scale = iw / LOGO_REFERENCE_WIDTH

    full_anchor = _load_gray_asset("baozimh_bottom_right_anchor.png")
    ys, xs = np.where(full_anchor > 0)
    if len(xs) == 0:
        return []
    ax0, ax1 = int(xs.min()), int(xs.max() + 1)
    ay0, ay1 = int(ys.min()), int(ys.max() + 1)
    anchor_crop = full_anchor[ay0:ay1, ax0:ax1]

    aw = max(1, round(anchor_crop.shape[1] * scale))
    ah = max(1, round(anchor_crop.shape[0] * scale))
    anchor_f = _resize_mask(anchor_crop, aw, ah).astype(np.float32) / 255.0
    yellow = _yellow_likelihood(rgb).astype(np.float32) / 255.0

    edge_offset = max(1, round(LOGO_EDGE_OFFSET_PX * scale))
    tolerance = max(6, round(LOGO_EDGE_TOLERANCE_PX * scale))
    threshold = min_confidence if min_confidence is not None else LOGO_MIN_CONFIDENCE

    box_w = max(1, round(LOGO_BOX_SIZE[0] * scale))
    box_h = max(1, round(LOGO_BOX_SIZE[1] * scale))

    results: list[Detection] = []
    windows = (
        ("top", max(0, edge_offset - tolerance), min(ih, edge_offset + tolerance + ah)),
        (
            "bottom",
            max(0, ih - (edge_offset + tolerance + ah)),
            min(ih, ih - (edge_offset - tolerance)),
        ),
    )

    logo_spec = TemplateSpec(
        name="baozimh_logo",
        anchor="top-left",
        reference_width=LOGO_REFERENCE_WIDTH,
        reference_height=None,
        box=(0, 0, LOGO_BOX_SIZE[0], LOGO_BOX_SIZE[1]),
        min_confidence=LOGO_MIN_CONFIDENCE,
        protect_dark=False,
        search_radius_px=0,
    )

    for edge, sy0, sy1 in windows:
        target = yellow[sy0:sy1, :]
        if target.shape[0] < ah or target.shape[1] < aw:
            continue

        # CCORR is deliberate here. CCOEFF can report spurious perfect matches
        # when a search window contains large constant/empty regions.
        confidence, peak_margin, max_loc = _template_peak_metrics(
            target,
            anchor_f,
            method=cv2.TM_CCORR_NORMED,
        )
        if confidence < threshold:
            continue

        anchor_x = int(max_loc[0])
        anchor_y = int(sy0 + max_loc[1])

        # A plain yellow patch can correlate surprisingly well with the rounded
        # square silhouette. The Baozimh icon also contains a stable dark face/
        # dumpling glyph, so validate the dark-pixel share inside the matched icon.
        # This rejects yellow clothing/flowers and solid yellow rectangles while
        # remaining tolerant of compression and the older logo rendering.
        icon_roi = rgb[anchor_y:anchor_y + ah, anchor_x:anchor_x + aw]
        if icon_roi.shape[:2] != (ah, aw):
            continue
        icon_gray = cv2.cvtColor(icon_roi, cv2.COLOR_RGB2GRAY)
        strict_valid = _logo_dark_glyph_valid(icon_gray)
        yellow_iou, yellow_recall = _yellow_shape_metrics(
            icon_roi,
            anchor_crop,
        )
        dark_pattern_correlation = _logo_dark_pattern_correlation(
            icon_roi,
            anchor_crop,
        )

        x = round(anchor_x - ax0 * scale)
        y = round(anchor_y - ay0 * scale)
        x = max(0, min(iw - box_w, x))
        y = max(0, min(ih - box_h, y))
        box = (int(x), int(y), int(box_w), int(box_h))

        (
            precise,
            variant,
            mask_variant,
            variant_support,
            competing_variant_support,
            variant_margin,
            variant_winner_gap,
            variant_offset,
        ) = _select_logo_variant(
            rgb,
            box,
            search_radius=max(2, round(8 * scale)),
        )
        joint_support = _logo_variant_joint_support(
            rgb[y:y + box_h, x:x + box_w],
            precise,
            full_anchor,
            yellow_recall=yellow_recall,
        ) if variant == "current" else 0.0

        glyph_mode = "strict"
        detection_reasons: list[str] = []
        if not strict_valid:
            # The strict validator remains unchanged and continues to protect
            # hard negatives.  The normal handoff still requires all
            # independent shape and pattern gates.  A second, narrower path
            # exists for the latest compact current logo: it requires an
            # unambiguous current variant and independent wordmark support, so
            # a current-looking speech bubble remains ambiguous/uncleaned.
            occlusion_consistent = (
                confidence >= LOGO_OCCLUSION_ANCHOR_GATE
                and peak_margin >= LOGO_OCCLUSION_PEAK_MARGIN_GATE
                and yellow_iou >= LOGO_OCCLUSION_YELLOW_IOU_GATE
                and yellow_recall >= LOGO_OCCLUSION_YELLOW_RECALL_GATE
                and dark_pattern_correlation >= LOGO_OCCLUSION_DARK_CORRELATION_GATE
            )
            current_variant_occlusion = (
                variant == "current"
                and confidence >= LOGO_MIN_CONFIDENCE
                and peak_margin >= LOGO_OCCLUSION_PEAK_MARGIN_GATE
                and yellow_iou >= LOGO_CURRENT_OCCLUSION_YELLOW_IOU_GATE
                and yellow_recall >= LOGO_CURRENT_OCCLUSION_YELLOW_RECALL_GATE
                and dark_pattern_correlation
                >= LOGO_CURRENT_OCCLUSION_DARK_CORRELATION_GATE
                and variant_support >= LOGO_CURRENT_OCCLUSION_VARIANT_SUPPORT_GATE
                and variant_margin >= 0.05
                and variant_winner_gap >= 0.04
                and joint_support >= LOGO_CURRENT_OCCLUSION_JOINT_SUPPORT_GATE
            )
            if not (occlusion_consistent or current_variant_occlusion):
                continue
            glyph_mode = "occlusion-consistent"
            detection_reasons.append("occluded_glyph")
            if current_variant_occlusion and not occlusion_consistent:
                detection_reasons.append("current_variant_joint_evidence")
        if variant == "ambiguous":
            detection_reasons.append("ambiguous_identity")
        anchor_retention = _logo_registration_anchor_retention(
            yellow, anchor_f, (anchor_x, anchor_y), variant_offset,
            max(1, round(2 * scale)),
        )
        if anchor_retention < 0.85:
            detection_reasons.extend(("uncertain_registration", "variant_anchor_conflict"))
        # The black dumpling face is part of the watermark, not protected comic
        # line art.  Template masks historically left holes in the yellow icon,
        # which is why a black/yellow icon outline survived on many cleaned
        # pages.  Once the distinctive icon has been validated, its complete
        # observed rectangle is safe to remove as one coherent object.
        icon_x0 = max(0, anchor_x - x - max(1, round(2 * scale)))
        icon_y0 = max(0, anchor_y - y - max(1, round(2 * scale)))
        icon_x1 = min(box_w, anchor_x - x + aw + max(1, round(2 * scale)))
        icon_y1 = min(box_h, anchor_y - y + ah + max(1, round(2 * scale)))
        precise[icon_y0:icon_y1, icon_x0:icon_x1] = 255
        if not aggressive:
            precise = _extend_logo_mask_palette_halo(
                rgb,
                box,
                precise,
                halo_px=max(2, round(3 * scale)),
            )
        mask = _build_aligned_mask(rgb, logo_spec, box, precise, aggressive, dilation)
        if not aggressive:
            # Generic dilation is useful for anti-aliased watermark edges, but it
            # must not grow onto nearby comic panel borders/line art. Preserve
            # very dark target pixels that are outside the intended logo support.
            # The dumpling face remains removable because it lies inside
            # ``precise`` itself.
            bx, by, bw, bh = box
            roi = rgb[by:by + bh, bx:bx + bw]
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            core = _resize_mask(precise, bw, bh) > 0
            structural_dark = ((gray < 115) & ~core).astype(np.uint8)
            structural_dark = cv2.dilate(
                structural_dark, np.ones((3, 3), np.uint8), iterations=1
            ) > 0
            local_mask = mask[by:by + bh, bx:bx + bw]
            local_mask[structural_dark] = 0

        if np.count_nonzero(mask) == 0:
            continue
        registration_confidence = max(
            0.0,
            min(
                1.0,
                0.85 * confidence
                + 0.15 * min(1.0, peak_margin / 0.10),
            ),
        )
        evidence = DetectionEvidence(
            family="modern-logo",
            variant=variant,
            mask_variant=mask_variant,
            edge=edge,
            scale=float(scale),
            registration_offset=(int(variant_offset[0]), int(variant_offset[1])),
            registration_uncertainty=(
                max(1, round(2 * scale)),
                max(1, round(2 * scale)),
            ),
            anchor_score=confidence,
            peak_margin=peak_margin,
            yellow_iou=yellow_iou,
            yellow_recall=yellow_recall,
            dark_pattern_correlation=dark_pattern_correlation,
            glyph_mode=glyph_mode,
            identity_confidence=confidence,
            registration_confidence=(
                min(registration_confidence, anchor_retention)
                if "variant_anchor_conflict" in detection_reasons
                else registration_confidence
            ),
            variant_support=variant_support,
            competing_variant_support=competing_variant_support,
            variant_margin=variant_margin,
            variant_winner_gap=variant_winner_gap,
            joint_support=joint_support,
            registration_anchor_retention=anchor_retention,
            reasons=tuple(detection_reasons),
        )
        results.append(Detection("baozimh_logo", confidence, box, mask, evidence))

    return results


def detect_watermarks(
    image: Image.Image | np.ndarray,
    *,
    aggressive: bool = False,
    dilation: int = 1,
    min_confidence: float | None = None,
) -> list[Detection]:
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image")

    detections: list[Detection] = _detect_edge_logos(
        rgb,
        aggressive=aggressive,
        dilation=dilation,
        min_confidence=min_confidence,
    )

    detections.extend(_detect_promo_blocks(rgb))
    detections.extend(
        _detect_legacy_banners(
            rgb,
            aggressive=aggressive,
            dilation=dilation,
            min_confidence=min_confidence,
        )
    )

    return detections


def combine_masks(detections: Iterable[Detection], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for d in detections:
        out = np.maximum(out, d.mask)
    return out


def combine_cleanup_masks(
    image: Image.Image | np.ndarray,
    detections: Iterable[Detection],
    *,
    logo_style: str = "solid",
    padding: int = 3,
    protect_dark: bool = True,
) -> np.ndarray:
    """Build the mask that should actually be passed to the inpainter.

    ``detect_watermarks`` deliberately returns a foreground-shaped mask because
    that is useful for inspection and keeps detection conservative.  Passing a
    character-shaped mask directly to an inpainting model, however, can make the
    model echo the shape of the removed Chinese text as faint smudges.

    For the modern Baozimh logo, ``solid`` therefore fills the tight bounding
    rectangle around the detected overlay before inpainting.  Very dark source
    structure outside the original watermark support (panel borders / line art)
    is punched back out of that rectangle so it is preserved exactly.

    ``precise`` retains the pre-0.2.6 behaviour.

    ``protect_dark=False`` keeps the solid region truly contiguous.  This is
    useful for high-overlap cases where a pale watermark sits on a dark area:
    punching the dark background back out can otherwise reduce the mask to
    character-shaped holes and cause LaMa to redraw the watermark text.
    """
    if logo_style not in {"solid", "precise"}:
        raise ValueError(f"Unsupported logo mask style: {logo_style}")

    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image")

    det_list = list(detections)
    out = combine_masks(det_list, rgb.shape[:2])
    if logo_style == "precise":
        return out

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ih, iw = gray.shape

    for d in det_list:
        if d.name != "baozimh_logo":
            continue

        core = d.mask > 0
        ys, xs = np.where(core)
        if len(xs) == 0:
            continue

        pad = max(0, int(padding))
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(iw, int(xs.max()) + 1 + pad)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(ih, int(ys.max()) + 1 + pad)

        local = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
        local_core = core[y0:y1, x0:x1]

        # Keep genuine black/dark comic structure outside the known watermark
        # support. Pixels coinciding with the logo template remain removable;
        # otherwise dark watermark glyphs can be mistaken for comic ink.
        if protect_dark:
            dark = ((gray[y0:y1, x0:x1] < 118) & ~local_core).astype(np.uint8)
            protected = cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=1) > 0
            protected &= ~local_core
            local[protected] = 0

            # Saturated comic lettering and effects (notably red chapter-title
            # calligraphy) are also real foreground.  Preserve them unless they
            # are yellow/orange pixels belonging to the Baozimh icon itself.
            local_hsv = hsv[y0:y1, x0:x1]
            non_yellow_color = (
                (local_hsv[:, :, 1] >= 65)
                & ~(
                    (local_hsv[:, :, 0] >= 7)
                    & (local_hsv[:, :, 0] <= 42)
                )
            )
            local[non_yellow_color] = 0
            out[y0:y1, x0:x1][non_yellow_color] = 0

        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], local)

    return out
