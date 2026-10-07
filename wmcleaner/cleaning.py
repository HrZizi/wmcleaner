from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .comic_overlap import max_structural_overlap_score
from .detector import (
    Detection,
    _logo_variant_mask,
    _observed_logo_icon_box,
    _resize_mask,
    combine_cleanup_masks,
    detect_watermarks,
)
from .inpaint import Inpainter
from .legacy_banner import (
    compose_legacy_banner_repair,
    prepare_legacy_banner_repair,
)
from .recovery_plan import (
    OverlayPlan,
    ResidualEvidence,
    ResidualCheck,
    build_overlay_plans,
    evaluate_registered_residual,
)


@dataclass(frozen=True)
class CleanSettings:
    backend: str = "auto"
    device: str = "auto"
    aggressive: bool = False
    dilation: int = 1
    min_confidence: float | None = None
    comic_overlap_backend: str = "auto"
    comic_overlap_threshold: float = 0.035
    comic_dense_overlap_threshold: float = 0.18
    logo_mask_style: str = "solid"
    logo_mask_padding: int = 3
    legacy_banner_mode: str = "auto"


@dataclass
class CleanImageResult:
    image: Image.Image
    mask: np.ndarray
    detections: list[Detection]
    cleaned: bool
    backend: str | None
    overlap_score: float
    used_comic_fallback: bool = False
    used_dense_logo_mask: bool = False
    dense_repair_mode: str | None = None
    used_legacy_banner_mask: bool = False
    legacy_banner_mode_used: str | None = None
    legacy_recovered_fraction: float = 0.0
    legacy_panel_pixel_count: int = 0
    legacy_deblended_pixel_count: int = 0
    legacy_deblended_fraction: float = 0.0
    legacy_ai_mask_pixel_count: int = 0
    legacy_ai_mask_fraction: float = 0.0
    legacy_structural_pixel_count: int = 0
    legacy_boundary_pixel_count: int = 0
    legacy_boundary_fraction: float = 0.0
    legacy_fit_space: str | None = None
    legacy_alpha: float | None = None
    legacy_fit_improvement: float = 0.0
    legacy_fallback_reason: str | None = None
    # Recovery-first metadata.  These fields are optional so callers that
    # construct compatibility results (for example pipeline tests) continue to
    # work unchanged.
    plans: tuple[OverlayPlan, ...] = ()
    qa_reasons: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)
    operation_mask: np.ndarray | None = None
    residual_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.provenance is None:
            self.provenance = {}

    @property
    def plan(self) -> tuple[OverlayPlan, ...]:
        """Singular compatibility alias for the per-overlay plan collection."""
        return self.plans

    @property
    def qa(self) -> tuple[str, ...]:
        return self.qa_reasons


def _prepare_dense_overlap_hybrid(
    image: Image.Image,
    detections: list[Detection],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Prepare a safer repair for watermark text spanning dark artwork.

    A fully solid 300-ish px watermark rectangle solved ghost text, but it also
    asked Anime-LaMa to recreate far too much real artwork (for example a skirt
    could become shorts).  For dense overlap, split the modern Baozimh logo:

    * the translucent gray text is removed with a tight, slightly dilated Telea
      mask; this changes only pixels that actually belong to the watermark;
    * the opaque yellow icon gets a small solid rectangle and is the only part
      handed to Anime-LaMa.

    The returned report mask is the union of both operations so saved masks / QA
    still describe every pixel intentionally repaired.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h_img, w_img = rgb.shape[:2]

    text_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    icon_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    for det in detections:
        if det.name != "baozimh_logo":
            continue

        x, y, w, h = det.box
        if w <= 0 or h <= 0:
            continue
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x >= x2 or y >= y2:
            continue

        roi = rgb[y:y2, x:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        yellow = (
            (hsv[:, :, 0] >= 7)
            & (hsv[:, :, 0] <= 42)
            & (hsv[:, :, 1] >= 35)
            & (hsv[:, :, 2] >= 110)
        ).astype(np.uint8)

        # The icon is the largest connected yellow component in the left half of
        # the aligned watermark box.  Using the actual observed icon rather than
        # a hard-coded pixel width keeps this scale-independent.
        left_limit = max(1, int(round(0.42 * roi.shape[1])))
        yellow[:, left_limit:] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(yellow, 8)
        icon_local: tuple[int, int, int, int] | None = None
        if count > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if int(stats[idx, cv2.CC_STAT_AREA]) >= max(20, int(0.002 * w * h)):
                ix = int(stats[idx, cv2.CC_STAT_LEFT])
                iy = int(stats[idx, cv2.CC_STAT_TOP])
                iw = int(stats[idx, cv2.CC_STAT_WIDTH])
                ih = int(stats[idx, cv2.CC_STAT_HEIGHT])
                pad = max(2, int(round(w / 160)))
                icon_local = (
                    max(0, ix - pad),
                    max(0, iy - pad),
                    min(roi.shape[1], ix + iw + pad),
                    min(roi.shape[0], iy + ih + pad),
                )

        precise = det.mask[y:y2, x:x2].copy()
        if icon_local is not None:
            ix0, iy0, ix1, iy1 = icon_local
            precise[iy0:iy1, ix0:ix1] = 0
            icon_mask[y + iy0:y + iy1, x + ix0:x + ix1] = 255
        else:
            # If icon localization ever fails, avoid the old destructive full
            # rectangle.  Leave the whole precise mask to the conservative path.
            pass

        text_mask[y:y2, x:x2] = np.maximum(text_mask[y:y2, x:x2], precise)

    if np.any(text_mask):
        text_mask = cv2.dilate(text_mask, np.ones((3, 3), np.uint8), iterations=1)
        cleaned_bgr = cv2.inpaint(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            text_mask,
            5,
            cv2.INPAINT_TELEA,
        )
        prepared = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
    else:
        prepared = rgb.copy()

    report_mask = cv2.bitwise_or(text_mask, icon_mask)
    # No localized icon means there is nothing safe to send to generative
    # inpainting.  A zero mask makes Inpainter preserve the Telea-only result.
    return Image.fromarray(prepared), icon_mask, report_mask


def _dense_logo_spans_dark_artwork(
    image: Image.Image,
    detections: list[Detection],
) -> bool:
    """Return whether a dense modern logo needs the dark-artwork hybrid.

    Telea is useful when the translucent logo text crosses a broad dark area:
    a precise generative mask can copy the pale lettering straight back into
    that area.  On light, detailed artwork the same Telea pass is harmful and
    can turn line art into repeated geometric smears.  Measure the aligned logo
    regions directly so those two cases take different repair paths.
    """
    gray = cv2.cvtColor(
        np.asarray(image.convert("RGB"), dtype=np.uint8),
        cv2.COLOR_RGB2GRAY,
    )
    dark_fractions: list[float] = []
    for detection in detections:
        if detection.name != "baozimh_logo":
            continue
        x, y, w, h = detection.box
        roi = gray[y:y + h, x:x + w]
        if roi.size:
            dark_fractions.append(float(np.mean(roi < 118)))
    return any(fraction >= 0.20 for fraction in dark_fractions)


def _modern_logo_scene_is_simple(
    image: Image.Image,
    detections: list[Detection],
) -> bool:
    """Return whether a coherent modern-logo redraw is safe for this scene.

    Protecting every dark pixel is useful over speech text and detailed food or
    clothing, but it fragments the mask on simple panel borders, flat garments,
    and dark stripes.  Those fragments are precisely where LaMa copied pieces
    of the Baozimh logo back into the result.  Prefer one Anime-LaMa region when
    the dark structure consists of only a few components, or one dominant flat
    dark area; retain the protected path for high-detail regions.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    decisions: list[bool] = []
    for detection in detections:
        if detection.name != "baozimh_logo":
            continue
        x, y, w, h = detection.box
        roi = rgb[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        dark = (gray < 100).astype(np.uint8)

        # Ignore the logo icon itself when deciding whether sparse dark ink is
        # real page structure.  Text, bubble arcs and panel borders outside the
        # icon must keep the protected path even if they form only a handful of
        # connected components.
        yellow = (
            (hsv[:, :, 0] >= 7)
            & (hsv[:, :, 0] <= 42)
            & (hsv[:, :, 1] >= 35)
            & (hsv[:, :, 2] >= 110)
        ).astype(np.uint8)
        yellow[:, max(1, int(round(0.42 * roi.shape[1]))):] = 0
        icon_count, _, icon_stats, _ = cv2.connectedComponentsWithStats(yellow, 8)
        if icon_count > 1:
            icon_index = 1 + int(np.argmax(icon_stats[1:, cv2.CC_STAT_AREA]))
            ix = int(icon_stats[icon_index, cv2.CC_STAT_LEFT])
            iy = int(icon_stats[icon_index, cv2.CC_STAT_TOP])
            iw = int(icon_stats[icon_index, cv2.CC_STAT_WIDTH])
            ih = int(icon_stats[icon_index, cv2.CC_STAT_HEIGHT])
            dark[max(0, iy - 2):min(h, iy + ih + 2), max(0, ix - 2):min(w, ix + iw + 2)] = 0

        non_yellow_color = (
            (hsv[:, :, 1] >= 65)
            & ~((hsv[:, :, 0] >= 7) & (hsv[:, :, 0] <= 42))
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        min_area = max(4, int(round(20 * (w / 320.0) ** 2)))
        areas = [
            int(stats[index, cv2.CC_STAT_AREA])
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= min_area
        ]
        component_count = len(areas)
        largest_fraction = max(areas, default=0) / dark.size
        sparse_ink_fraction = float(np.mean(dark > 0))
        colored_structure_fraction = float(np.mean(non_yellow_color))
        decisions.append(
            largest_fraction >= 0.14
            or (
                component_count <= 9
                and sparse_ink_fraction < 0.006
                and colored_structure_fraction < 0.006
            )
        )
    return bool(decisions) and all(decisions)


def _legacy_banner_has_uniform_pale_background(
    image: Image.Image,
    detection: Detection,
) -> bool:
    """Return whether a legacy banner can be replaced by its pale margin."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    x, y, w, h = detection.box
    pad = max(8, round(12 * min(w / 325.0, h / 92.0)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(rgb.shape[1], x + w + pad)
    y1 = min(rgb.shape[0], y + h + pad)
    region = rgb[y0:y1, x0:x1]
    ring = np.ones(region.shape[:2], dtype=bool)
    ring[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
    samples = region[ring]
    return bool(len(samples)) and float(np.mean(samples.min(axis=1) > 235)) >= 0.97


def _fill_uniform_legacy_banners(
    image: Image.Image,
    detections: list[Detection],
) -> tuple[Image.Image, np.ndarray]:
    """Remove legacy banners on empty pale margins without hallucinating."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for detection in detections:
        x, y, w, h = detection.box
        x2 = min(rgb.shape[1], x + w)
        y2 = min(rgb.shape[0], y + h)
        pad = max(8, round(12 * min(w / 325.0, h / 92.0)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x3 = min(rgb.shape[1], x2 + pad)
        y3 = min(rgb.shape[0], y2 + pad)
        region = rgb[y0:y3, x0:x3]
        ring = np.ones(region.shape[:2], dtype=bool)
        ring[y - y0:y2 - y0, x - x0:x2 - x0] = False
        samples = region[ring]
        pale = samples[samples.min(axis=1) > 235]
        color = (
            np.median(pale, axis=0).astype(np.uint8)
            if len(pale)
            else np.array([255, 255, 255], dtype=np.uint8)
        )
        local = rgb[y:y2, x:x2]
        # A pale margin can still contain a speech-bubble arc or panel border.
        # Preserve that observed ink exactly; the legacy banner's hands/text
        # are yellow or pale gray, so they remain safely replaceable.
        structural = cv2.cvtColor(local, cv2.COLOR_RGB2GRAY) < 100
        local[~structural] = color
        mask[y:y2, x:x2][~structural] = 255
        _restore_circular_outline(
            np.asarray(image.convert("RGB"), dtype=np.uint8),
            rgb,
            mask,
            detection,
        )
    return Image.fromarray(rgb), mask


def _restore_circular_outline(
    source: np.ndarray,
    result: np.ndarray,
    repair_mask: np.ndarray,
    detection: Detection,
    *,
    min_support: float = 0.16,
) -> None:
    """Bridge a watermark-sized gap in a strongly supported bubble circle."""
    x, y, w, h = detection.box
    pad_x = max(140, round(0.8 * w))
    pad_y = max(100, round(1.8 * h))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(source.shape[1], x + w + pad_x)
    y1 = min(source.shape[0], y + h + pad_y)
    gray = cv2.cvtColor(source[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    if gray.size == 0 or float(np.mean(gray < 105)) < 0.002:
        return
    dx0 = x - x0
    dy0 = y - y0
    detection_gray = gray[dy0:dy0 + h, dx0:dx0 + w]
    detection_edges = cv2.Canny(detection_gray, 60, 150)
    straight = cv2.HoughLinesP(
        detection_edges,
        1,
        np.pi / 180,
        threshold=max(30, round(0.18 * w)),
        minLineLength=max(45, round(0.42 * w)),
        maxLineGap=8,
    )
    if straight is not None:
        horizontal_count = 0
        vertical_count = 0
        for x_a, y_a, x_b, y_b in straight[:, 0]:
            horizontal_count += int(abs(y_b - y_a) <= 2)
            vertical_count += int(abs(x_b - x_a) <= 2)
        # A double/parallel panel border is decisive.  A bubble's shallow top
        # or bottom arc can look like one straight segment at this scale, so a
        # lone Hough line must not suppress circular reconstruction.
        if horizontal_count >= 2 or vertical_count >= 2:
            return
    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(60, round(0.25 * w)),
        param1=100,
        param2=55,
        minRadius=max(45, round(0.28 * w)),
        maxRadius=max(90, round(1.35 * w)),
    )
    if circles is None:
        return

    local_repair = repair_mask[y0:y1, x0:x1] > 0
    best: tuple[float, int, int, int, np.ndarray] | None = None
    thickness = max(2, round(3 * w / 320.0))
    for cx, cy, radius in np.round(circles[0]).astype(int):
        ring = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(ring, (cx, cy), radius, 255, thickness + 2)
        ring_pixels = ring > 0
        missing = ring_pixels & local_repair
        supported = ring_pixels & ~local_repair
        if int(np.count_nonzero(missing)) < max(8, thickness * 3):
            continue
        support_count = int(np.count_nonzero(supported))
        if support_count == 0:
            continue
        support = float(np.mean(gray[supported] < 105))
        if support < min_support:
            continue
        if best is None or support > best[0]:
            best = (support, cx, cy, radius, supported)
    if best is None:
        return

    _, cx, cy, radius, supported = best
    # Hough circles are quantized and can land several pixels away from the
    # surviving bubble stroke. Refine the winner against only observed dark
    # ink outside the repair area. This prevents the reconstructed section
    # from meeting the real arc as a notch or a parallel/doubled line.
    visible_ink = (gray < 105) & ~local_repair
    outline_ellipse: (
        tuple[tuple[float, float], tuple[float, float], float] | None
    ) = None
    outline_polyline: np.ndarray | None = None
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(visible_ink.astype(np.uint8), 8)
    )
    grid_y, grid_x = np.indices(gray.shape)
    candidate_band = (
        np.abs(np.hypot(grid_x - cx, grid_y - cy) - radius)
        <= max(14, round(0.1 * radius))
    )
    arc_components: list[int] = []
    component_overlap = 0
    for index in range(1, component_count):
        stats = component_stats[index]
        if (
            int(stats[cv2.CC_STAT_AREA]) < 24
            or max(
                int(stats[cv2.CC_STAT_WIDTH]),
                int(stats[cv2.CC_STAT_HEIGHT]),
            )
            < 0.18 * radius
        ):
            continue
        overlap = int(
            np.count_nonzero((component_labels == index) & candidate_band)
        )
        if overlap >= 12:
            arc_components.append(index)
            component_overlap += overlap

    if arc_components and component_overlap >= 24:
        fit_y, fit_x = np.nonzero(np.isin(component_labels, arc_components))
        fit_x = fit_x.astype(np.float64)
        fit_y = fit_y.astype(np.float64)
        original_circle = (cx, cy, radius)
        refined: tuple[float, float, float] | None = None
        design = np.column_stack((2.0 * fit_x, 2.0 * fit_y, np.ones(len(fit_x))))
        values = fit_x * fit_x + fit_y * fit_y
        try:
            fit_cx, fit_cy, offset = np.linalg.lstsq(
                design,
                values,
                rcond=None,
            )[0]
        except np.linalg.LinAlgError:
            pass
        else:
            squared_radius = offset + fit_cx * fit_cx + fit_cy * fit_cy
            if squared_radius > 0:
                refined = (
                    float(fit_cx),
                    float(fit_cy),
                    float(np.sqrt(squared_radius)),
                )

        if len(fit_x) >= 24 and refined is not None:
            fit_cx, fit_cy, fit_radius = refined
            shift = float(
                np.hypot(
                    fit_cx - original_circle[0],
                    fit_cy - original_circle[1],
                )
            )
            radius_shift = abs(fit_radius - original_circle[2])
            if shift <= max(12.0, 0.18 * original_circle[2]) and radius_shift <= max(
                10.0,
                0.16 * original_circle[2],
            ):
                cx = int(round(fit_cx))
                cy = int(round(fit_cy))
                radius = int(round(fit_radius))
                refined_ring = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(refined_ring, (cx, cy), radius, 255, thickness + 2)
                supported = (refined_ring > 0) & ~local_repair

                component_points = np.column_stack((fit_x, fit_y)).astype(np.float32)
                ellipse = cv2.fitEllipse(component_points.reshape(-1, 1, 2))
                (ellipse_cx, ellipse_cy), (axis_a, axis_b), ellipse_angle = ellipse
                mean_radius = 0.25 * (axis_a + axis_b)
                ellipse_shift = float(np.hypot(ellipse_cx - cx, ellipse_cy - cy))
                if (
                    min(axis_a, axis_b) >= 1.45 * radius
                    and max(axis_a, axis_b) <= 2.55 * radius
                    and max(axis_a, axis_b) / min(axis_a, axis_b) <= 1.25
                    and ellipse_shift <= max(10.0, 0.1 * radius)
                    and abs(mean_radius - radius) <= max(8.0, 0.08 * radius)
                ):
                    outline_ellipse = (
                        (float(ellipse_cx), float(ellipse_cy)),
                        (float(axis_a), float(axis_b)),
                        float(ellipse_angle),
                    )
                    ellipse_ring = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.ellipse(
                        ellipse_ring,
                        outline_ellipse,
                        255,
                        thickness + 2,
                        lineType=cv2.LINE_AA,
                    )
                    supported = (ellipse_ring > 0) & ~local_repair

        # A wide banner can split the top of a speech bubble into two
        # disconnected side arcs. A global contour fit is poorly conditioned
        # in that layout. Warp the supported Hough arc onto the two observed
        # endpoints with a cubic correction constrained by the observed local
        # slopes. This makes both position and tangent continuous at the seams.
        horizontal_parts: list[tuple[np.ndarray, np.ndarray]] = []
        for index in arc_components:
            part_y, part_x = np.nonzero(component_labels == index)
            unique_x = np.unique(part_x)
            if len(unique_x) < max(12, round(0.12 * radius)):
                continue
            center_y = np.array(
                [float(np.median(part_y[part_x == value])) for value in unique_x],
                dtype=np.float64,
            )
            horizontal_parts.append((unique_x.astype(np.float64), center_y))

        if len(horizontal_parts) >= 2:
            horizontal_parts.sort(key=lambda part: float(np.mean(part[0])))
            left_x, left_y = horizontal_parts[0]
            right_x, right_y = horizontal_parts[-1]
            start_x = float(left_x[-1])
            end_x = float(right_x[0])
            gap_width = end_x - start_x
            if gap_width >= max(24.0, 0.35 * radius):
                start_y = float(np.median(left_y[-3:]))
                end_y = float(np.median(right_y[:3]))
                blend_span = max(8.0, min(16.0, 0.08 * radius))
                left_anchor_index = int(
                    np.argmin(np.abs(left_x - (start_x - blend_span)))
                )
                right_anchor_index = int(
                    np.argmin(np.abs(right_x - (end_x + blend_span)))
                )
                left_anchor_x = float(left_x[left_anchor_index])
                left_anchor_y = float(left_y[left_anchor_index])
                right_anchor_x = float(right_x[right_anchor_index])
                right_anchor_y = float(right_y[right_anchor_index])
                sample_x = np.arange(
                    int(np.floor(left_anchor_x)),
                    int(np.ceil(right_anchor_x)) + 1,
                    dtype=np.float64,
                )
                radial = np.sqrt(
                    np.maximum(0.0, radius * radius - (sample_x - cx) ** 2)
                )
                upper_y = cy - radial
                lower_y = cy + radial
                start_radial = float(
                    np.sqrt(max(0.0, radius * radius - (start_x - cx) ** 2))
                )
                end_radial = float(
                    np.sqrt(max(0.0, radius * radius - (end_x - cx) ** 2))
                )
                upper_start = cy - start_radial
                upper_end = cy - end_radial
                lower_start = cy + start_radial
                lower_end = cy + end_radial
                upper_error = abs(upper_start - start_y) + abs(upper_end - end_y)
                lower_error = abs(lower_start - start_y) + abs(lower_end - end_y)
                sample_y = upper_y if upper_error <= lower_error else lower_y
                branch_start = upper_start if upper_error <= lower_error else lower_start
                branch_end = upper_end if upper_error <= lower_error else lower_end
                start_offset = start_y - branch_start
                end_offset = end_y - branch_end
                tangent_samples = 30
                left_slope = float(
                    np.polyfit(
                        left_x[-tangent_samples:],
                        left_y[-tangent_samples:],
                        1,
                    )[0]
                )
                right_slope = float(
                    np.polyfit(
                        right_x[:tangent_samples],
                        right_y[:tangent_samples],
                        1,
                    )[0]
                )
                branch_sign = 1.0 if upper_error <= lower_error else -1.0
                start_base_slope = branch_sign * (start_x - cx) / max(
                    start_radial,
                    1.0,
                )
                end_base_slope = branch_sign * (end_x - cx) / max(
                    end_radial,
                    1.0,
                )
                left_anchor_radial = float(
                    np.sqrt(
                        max(
                            0.0,
                            radius * radius - (left_anchor_x - cx) ** 2,
                        )
                    )
                )
                right_anchor_radial = float(
                    np.sqrt(
                        max(
                            0.0,
                            radius * radius - (right_anchor_x - cx) ** 2,
                        )
                    )
                )
                left_anchor_branch = (
                    cy - left_anchor_radial
                    if upper_error <= lower_error
                    else cy + left_anchor_radial
                )
                right_anchor_branch = (
                    cy - right_anchor_radial
                    if upper_error <= lower_error
                    else cy + right_anchor_radial
                )
                left_anchor_offset = left_anchor_y - left_anchor_branch
                right_anchor_offset = right_anchor_y - right_anchor_branch
                left_anchor_base_slope = branch_sign * (left_anchor_x - cx) / max(
                    left_anchor_radial,
                    1.0,
                )
                right_anchor_base_slope = branch_sign * (
                    right_anchor_x - cx
                ) / max(right_anchor_radial, 1.0)
                start_slope_offset = float(
                    np.clip(left_slope - start_base_slope, -0.35, 0.35)
                )
                end_slope_offset = float(
                    np.clip(right_slope - end_base_slope, -0.35, 0.35)
                )
                left_anchor_slope_offset = float(
                    np.clip(left_slope - left_anchor_base_slope, -0.35, 0.35)
                )
                right_anchor_slope_offset = float(
                    np.clip(right_slope - right_anchor_base_slope, -0.35, 0.35)
                )
                correction = np.zeros_like(sample_x)
                middle = (sample_x >= start_x) & (sample_x <= end_x)
                t = np.clip((sample_x[middle] - start_x) / gap_width, 0.0, 1.0)
                correction[middle] = (
                    (2.0 * t**3 - 3.0 * t**2 + 1.0) * start_offset
                    + (t**3 - 2.0 * t**2 + t)
                    * gap_width
                    * start_slope_offset
                    + (-2.0 * t**3 + 3.0 * t**2) * end_offset
                    + (t**3 - t**2) * gap_width * end_slope_offset
                )
                left_blend = (sample_x < start_x) & (sample_x >= left_anchor_x)
                left_width = max(1.0, start_x - left_anchor_x)
                left_t = np.clip(
                    (sample_x[left_blend] - left_anchor_x) / left_width,
                    0.0,
                    1.0,
                )
                correction[left_blend] = (
                    (2.0 * left_t**3 - 3.0 * left_t**2 + 1.0)
                    * left_anchor_offset
                    + (left_t**3 - 2.0 * left_t**2 + left_t)
                    * left_width
                    * left_anchor_slope_offset
                    + (-2.0 * left_t**3 + 3.0 * left_t**2) * start_offset
                    + (left_t**3 - left_t**2) * left_width * start_slope_offset
                )
                right_blend = (sample_x > end_x) & (
                    sample_x <= right_anchor_x
                )
                right_width = max(1.0, right_anchor_x - end_x)
                right_t = np.clip(
                    (sample_x[right_blend] - end_x) / right_width,
                    0.0,
                    1.0,
                )
                correction[right_blend] = (
                    (2.0 * right_t**3 - 3.0 * right_t**2 + 1.0)
                    * end_offset
                    + (right_t**3 - 2.0 * right_t**2 + right_t)
                    * right_width
                    * end_slope_offset
                    + (-2.0 * right_t**3 + 3.0 * right_t**2)
                    * right_anchor_offset
                    + (right_t**3 - right_t**2)
                    * right_width
                    * right_anchor_slope_offset
                )
                sample_y = sample_y + correction
                valid = (
                    (sample_x >= 0)
                    & (sample_x < gray.shape[1])
                    & (sample_y >= 0)
                    & (sample_y < gray.shape[0])
                )
                if int(np.count_nonzero(valid)) >= 20:
                    outline_polyline = np.column_stack(
                        (sample_x[valid], sample_y[valid])
                    ).round().astype(np.int32)
                    polyline_ring = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.polylines(
                        polyline_ring,
                        [outline_polyline],
                        False,
                        255,
                        thickness + 2,
                        lineType=cv2.LINE_AA,
                    )
                    supported = (polyline_ring > 0) & ~local_repair

    dark_samples = source[y0:y1, x0:x1][supported & (gray < 105)]
    color = (
        tuple(int(v) for v in np.median(dark_samples, axis=0))
        if len(dark_samples)
        else (24, 24, 24)
    )
    rendered = result[y0:y1, x0:x1].copy()
    outline_line = np.zeros(gray.shape, dtype=np.uint8)
    if outline_polyline is not None:
        bridge_thickness = max(1, thickness - 1)
        cv2.polylines(
            rendered,
            [outline_polyline],
            False,
            color,
            bridge_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.polylines(
            outline_line,
            [outline_polyline],
            False,
            255,
            bridge_thickness + 1,
            lineType=cv2.LINE_AA,
        )
    elif outline_ellipse is not None:
        cv2.ellipse(
            rendered,
            outline_ellipse,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.ellipse(
            outline_line,
            outline_ellipse,
            255,
            thickness + 1,
            lineType=cv2.LINE_AA,
        )
    else:
        cv2.circle(
            rendered,
            (cx, cy),
            radius,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(outline_line, (cx, cy), radius, 255, thickness + 1)
    repair_support = cv2.dilate(
        local_repair.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ) > 0
    if outline_polyline is not None:
        seam_support = cv2.dilate(
            local_repair.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
            iterations=1,
        ) > 0
        corridor = cv2.dilate(
            outline_line,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        ) > 0
        corridor &= seam_support
        first_column = int(np.min(outline_polyline[:, 0]))
        last_column = int(np.max(outline_polyline[:, 0]))
        corridor &= (grid_x >= first_column) & (grid_x <= last_column)
        local_result = result[y0:y1, x0:x1]
        pale_samples = local_result[corridor & (local_result.min(axis=2) > 220)]
        if len(pale_samples):
            local_result[corridor] = np.median(pale_samples, axis=0).astype(np.uint8)
        repair_support = seam_support
    apply = (outline_line > 0) & repair_support
    result[y0:y1, x0:x1][apply] = rendered[apply]


def _bridge_sparse_straight_strokes(
    result: np.ndarray,
    repair_mask: np.ndarray,
    detection: Detection,
) -> None:
    """Join short collinear gaps in sparse line art inside a modern mark."""
    x, y, w, h = detection.box
    target = result[y:y + h, x:x + w]
    local_repair = repair_mask[y:y + h, x:x + w] > 0
    if target.size == 0 or not np.any(local_repair):
        return

    gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    dark = gray < 105
    dark_fraction = float(np.mean(dark))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark.astype(np.uint8),
        8,
    )
    components = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= 4
    ]
    # This path is intentionally limited to sparse emphasis/panel strokes.
    # Text-dense bubbles and single curves have dedicated handling.
    if not (0.0015 <= dark_fraction < 0.012 and 4 <= len(components) <= 12):
        return

    fitted: list[tuple[int, np.ndarray, np.ndarray, float, float]] = []
    for index in components:
        points_y, points_x = np.nonzero(labels == index)
        if len(points_x) < 4:
            continue
        points = np.column_stack((points_x, points_y)).astype(np.float64)
        center = np.mean(points, axis=0)
        covariance = np.cov(points - center, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[-1] < 0.8 or eigenvalues[-1] < 3.0 * max(
            eigenvalues[0],
            0.15,
        ):
            continue
        direction = eigenvectors[:, -1]
        if direction[0] < 0:
            direction = -direction
        projection = (points - center) @ direction
        fitted.append(
            (
                index,
                center,
                direction,
                float(np.min(projection)),
                float(np.max(projection)),
            )
        )

    predicted = np.zeros(dark.shape, dtype=np.uint8)
    for left_pos, left in enumerate(fitted):
        _, center_a, direction_a, min_a, max_a = left
        for right in fitted[left_pos + 1:]:
            _, center_b, direction_b, min_b, max_b = right
            alignment = abs(float(np.dot(direction_a, direction_b)))
            if alignment < float(np.cos(np.deg2rad(12.0))):
                continue
            direction = (
                direction_a
                + np.sign(np.dot(direction_a, direction_b)) * direction_b
            )
            direction /= np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]])
            if abs(float(np.dot(center_b - center_a, normal))) > 3.0:
                continue

            points_a = np.array(
                [center_a + min_a * direction_a, center_a + max_a * direction_a]
            )
            points_b = np.array(
                [center_b + min_b * direction_b, center_b + max_b * direction_b]
            )
            distances = np.linalg.norm(
                points_a[:, None, :] - points_b[None, :, :],
                axis=2,
            )
            endpoint_a, endpoint_b = np.unravel_index(
                np.argmin(distances),
                distances.shape,
            )
            gap = float(distances[endpoint_a, endpoint_b])
            if not (2.0 <= gap <= 20.0):
                continue
            start = tuple(np.round(points_a[endpoint_a]).astype(int))
            end = tuple(np.round(points_b[endpoint_b]).astype(int))
            cv2.line(predicted, start, end, 255, 1, lineType=cv2.LINE_AA)

    apply = (predicted > 0) & local_repair
    if not np.any(apply):
        return
    ink = np.median(target[dark], axis=0).astype(np.uint8)
    target[apply] = ink
    repair_mask[y:y + h, x:x + w][apply] = 0


def _modern_logo_has_uniform_pale_background(
    image: Image.Image,
    detection: Detection,
) -> bool:
    """Return whether a modern logo has a validated pale surface.

    This compatibility predicate now delegates to the same held-out surface
    evidence used by the runtime planner.  Geometry callers still receive the
    historical answer for supported card-like artwork; unsupported circular
    evidence remains a rejection.
    """
    plan, _ = build_overlay_plans(image, [detection])
    if not plan:
        return False
    selected = plan[0]
    # Keep the helper's historical compatibility behavior for a sparse curve
    # wholly inside the aligned box.  It is not used by the default runtime
    # router (which has the planner's connected-structure mask), but callers of
    # this private predicate rely on it to avoid deterministic flattening.
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    x, y, w, h = detection.box
    local = rgb[y:y + h, x:x + w]
    if local.size:
        gray = cv2.cvtColor(local, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(local, cv2.COLOR_RGB2HSV)
        core = detection.mask[y:y + h, x:x + w] > 0
        dark = (gray < 105) & ~core
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            dark.astype(np.uint8),
            8,
        )
        significant = sum(
            int(stats[index, cv2.CC_STAT_AREA]) >= 5
            for index in range(1, count)
        )
        colored = (
            (hsv[:, :, 1] >= 65)
            & ~((hsv[:, :, 0] >= 7) & (hsv[:, :, 0] <= 42))
            & ~core
        )
        if (
            0.0015 <= float(np.mean(dark)) < 0.012
            and significant <= 3
            and float(np.mean(colored)) < 0.004
        ):
            return False
    return bool(
        selected.surface.valid
        and selected.scene.geometry_kind != "circle"
    )


def _fill_uniform_modern_logos(
    image: Image.Image,
    detections: list[Detection],
) -> tuple[Image.Image, np.ndarray]:
    """Remove modern logos on pale regions without inventing text or borders."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for detection in detections:
        x, y, w, h = detection.box
        x2 = min(rgb.shape[1], x + w)
        y2 = min(rgb.shape[0], y + h)
        pad = max(10, round(16 * min(w / 320.0, h / 110.0)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x3 = min(rgb.shape[1], x2 + pad)
        y3 = min(rgb.shape[0], y2 + pad)
        region = original[y0:y3, x0:x3]
        ring = np.ones(region.shape[:2], dtype=bool)
        ring[y - y0:y2 - y0, x - x0:x2 - x0] = False
        samples = region[ring]
        pale = samples[samples.min(axis=1) > 230]
        color = np.median(pale, axis=0).astype(np.uint8)

        local = original[y:y2, x:x2]
        gray = cv2.cvtColor(local, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(local, cv2.COLOR_RGB2HSV)
        # The modern wordmark is pale gray, while comic ink is genuinely
        # dark. Preserve that observed ink even when it crosses the template
        # mask; the only dark part of the watermark itself is the icon, which
        # is removed explicitly below. This keeps thin rays, bubble outlines,
        # and lettering intact instead of asking a geometry heuristic to
        # recreate pixels that are already visible in the source.
        structural = gray < 118
        non_yellow_color = (
            (hsv[:, :, 1] >= 65)
            & ~((hsv[:, :, 0] >= 7) & (hsv[:, :, 0] <= 42))
        )
        icon_box = _observed_logo_icon_box(original, detection.box)
        if icon_box is not None:
            ix0, iy0, ix1, iy1 = icon_box
            structural[iy0:iy1, ix0:ix1] = False
            non_yellow_color[iy0:iy1, ix0:ix1] = False
        preserve = structural | non_yellow_color
        target = rgb[y:y2, x:x2]
        target[~preserve] = color
        target[preserve] = local[preserve]
        mask[y:y2, x:x2][~preserve] = 255

        # Bridge panel borders when dark evidence spans most of a row or
        # column.  This also handles a core/text gap between the observed ink
        # and the icon; isolated character strokes are too short and sparse to
        # satisfy both guards.
        dark = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY) < 105
        ink = (
            np.median(local[dark], axis=0).astype(np.uint8)
            if np.any(dark)
            else np.array([24, 24, 24], dtype=np.uint8)
        )
        min_horizontal_span = max(60, round(0.58 * target.shape[1]))
        min_horizontal_ink = max(25, round(0.24 * target.shape[1]))
        for row in range(target.shape[0]):
            positions = np.flatnonzero(dark[row])
            if (
                len(positions) >= min_horizontal_ink
                and int(positions[-1] - positions[0] + 1) >= min_horizontal_span
            ):
                target[row, positions[0]:positions[-1] + 1] = ink
                mask[y + row, x + positions[0]:x + positions[-1] + 1] = 0

        min_vertical_span = max(40, round(0.58 * target.shape[0]))
        min_vertical_ink = max(18, round(0.28 * target.shape[0]))
        for col in range(target.shape[1]):
            positions = np.flatnonzero(dark[:, col])
            if (
                len(positions) >= min_vertical_ink
                and int(positions[-1] - positions[0] + 1) >= min_vertical_span
            ):
                target[positions[0]:positions[-1] + 1, col] = ink
                mask[y + positions[0]:y + positions[-1] + 1, x + col] = 0
        _bridge_sparse_straight_strokes(rgb, mask, detection)
        # Modern marks sit over a wider range of edge shapes than legacy
        # banners. Require stronger observed-circle support so a curved panel
        # edge cannot be promoted into a false speech bubble.
        _restore_circular_outline(
            original,
            rgb,
            mask,
            detection,
            min_support=0.23,
        )
    return Image.fromarray(rgb), mask


def _polish_modern_logo_text(
    result: Image.Image,
    source: Image.Image,
    detections: list[Detection],
    runner: Inpainter | None = None,
) -> Image.Image:
    """Remove pale text fragments without regenerating the opaque icon area."""
    _, icon_mask, report_mask = _prepare_dense_overlap_hybrid(source, detections)
    text_mask = report_mask.copy()
    if np.any(icon_mask):
        icon_guard = cv2.dilate(
            icon_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        ) > 0
        text_mask[icon_guard] = 0
    if not np.any(text_mask):
        return result
    rgb = np.asarray(result.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    local_median_small = cv2.medianBlur(gray, 15)
    local_saturation_small = cv2.medianBlur(hsv[:, :, 1], 15)
    direct_candidate = (
        (text_mask > 0)
        & (
            (
                (hsv[:, :, 1] <= 18)
                & (gray >= 105)
                & (gray <= 235)
            )
            | (
                hsv[:, :, 1].astype(np.int16)
                <= local_saturation_small.astype(np.int16) - 6
            )
        )
        & (
            gray.astype(np.int16) - local_median_small.astype(np.int16)
            >= 12
        )
    )

    local_median = cv2.medianBlur(gray, 31)
    local_saturation = cv2.medianBlur(hsv[:, :, 1], 31)
    neutral_mid_tone = (
        (hsv[:, :, 1] <= 22)
        & (gray >= 95)
        & (gray <= 240)
    )
    desaturated_light = (
        hsv[:, :, 1].astype(np.int16)
        <= local_saturation.astype(np.int16) - 4
    )
    lightening = gray.astype(np.int16) - local_median.astype(np.int16)
    candidate = (
        (text_mask > 0)
        & (neutral_mid_tone | desaturated_light)
        & (lightening >= 8)
    )
    if not np.any(candidate):
        return result

    # A broad local mask can damage hands, clothing and panel borders.  Use the
    # candidate pixels only as evidence that one exact template component has
    # survived, then remove that component's original footprint.  Requiring a
    # substantial covered fraction rejects coincidental highlights and line art.
    source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    residual = np.zeros(gray.shape, dtype=np.uint8)
    for detection in detections:
        if detection.name != "baozimh_logo":
            continue
        x, y, w, h = detection.box
        template = _resize_mask(
            _logo_variant_mask(source_rgb, detection.box),
            w,
            h,
        )
        icon_box = _observed_logo_icon_box(source_rgb, detection.box)
        if icon_box is not None:
            ix0, iy0, ix1, iy1 = icon_box
            template[iy0:iy1, ix0:ix1] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (template > 0).astype(np.uint8),
            8,
        )
        local_candidate = candidate[y:y + h, x:x + w]
        local_residual = residual[y:y + h, x:x + w]
        for index in range(1, count):
            component = labels == index
            area = int(stats[index, cv2.CC_STAT_AREA])
            hits = int(np.count_nonzero(local_candidate[component]))
            if area and hits / area >= 0.10:
                local_residual[component] = 255
    if not np.any(residual):
        return result
    if runner is not None:
        polish_mask = cv2.dilate(
            residual,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        return runner.run(result, Image.fromarray(polish_mask))
    # Without a model runner, first remove only pixels with direct evidence so
    # the component-level pass receives clean boundary colors to propagate.
    prepolished_bgr = cv2.inpaint(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        direct_candidate.astype(np.uint8) * 255,
        2,
        cv2.INPAINT_TELEA,
    )
    repaired = cv2.inpaint(
        prepolished_bgr,
        residual,
        2,
        cv2.INPAINT_TELEA,
    )
    return Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))


def _coherent_legacy_mask(
    image: Image.Image,
    detections: list[Detection],
    *,
    padding: int,
) -> np.ndarray:
    """Build expanded rectangular masks for coherent legacy reconstruction."""
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for detection in detections:
        if detection.name != "baozimh_top_left":
            continue
        x, y, w, h = detection.box
        scale = min(w / 325.0, h / 92.0)
        pad = max(0, round(padding * scale))
        mask[
            max(0, y - pad):min(image.height, y + h + pad),
            max(0, x - pad):min(image.width, x + w + pad),
        ] = 255
    return mask


def _legacy_banner_overlaps_clipped_bubble(
    image: Image.Image,
    detections: list[Detection],
) -> bool:
    """Detect a top-clipped pale bubble crossing a dark legacy-banner scene."""
    gray = cv2.cvtColor(
        np.asarray(image.convert("RGB"), dtype=np.uint8),
        cv2.COLOR_RGB2GRAY,
    )
    for detection in detections:
        if detection.name != "baozimh_top_left":
            continue
        x, y, w, h = detection.box
        if y > 25 or x >= image.width // 2:
            continue
        roi = gray[y:y + h, x:x + w]
        if roi.size and float(np.mean(roi < 92)) >= 0.05:
            return True
    return False


def _compose_clipped_bubble_repair(
    inside: Image.Image,
    outside: Image.Image,
    detection: Detection,
) -> Image.Image:
    """Combine clean background with the pale bubble inferred by Anime-LaMa."""
    inside_rgb = np.asarray(inside.convert("RGB"), dtype=np.uint8)
    result = np.asarray(outside.convert("RGB"), dtype=np.uint8).copy()
    x, y, w, h = detection.box
    search_x2 = min(inside.width, x + w + round(85 * w / 325.0))
    search_y2 = min(inside.height, y + h + round(38 * h / 92.0))
    gray = cv2.cvtColor(inside_rgb[:search_y2, :search_x2], cv2.COLOR_RGB2GRAY)
    pale = (gray > 180).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pale, 8)
    candidates = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_TOP]) == 0
        and int(stats[index, cv2.CC_STAT_LEFT]) < x + w
        and int(stats[index, cv2.CC_STAT_AREA]) >= max(500, int(0.02 * w * h))
    ]
    if not candidates:
        return inside.convert("RGB")

    index = max(candidates, key=lambda value: int(stats[value, cv2.CC_STAT_AREA]))
    component = (labels == index).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return inside.convert("RGB")
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    polygon = cv2.approxPolyDP(hull, 0.025 * cv2.arcLength(hull, True), True)
    if len(polygon) < 4:
        return inside.convert("RGB")

    bubble_mask = np.zeros(result.shape[:2], dtype=np.uint8)
    cv2.fillPoly(bubble_mask, [polygon], 255)
    result[bubble_mask > 0] = inside_rgb[bubble_mask > 0]
    thickness = max(2, round(3 * min(w / 325.0, h / 92.0)))
    cv2.polylines(
        result,
        [polygon],
        True,
        (18, 18, 22),
        thickness,
        lineType=cv2.LINE_AA,
    )
    result[:thickness] = inside_rgb[:thickness]
    return Image.fromarray(result)


def _repair_clipped_bubble_banner(
    image: Image.Image,
    detections: list[Detection],
    runner: Inpainter,
) -> tuple[Image.Image, np.ndarray]:
    """Reconstruct a banner-obscured clipped bubble and its exterior separately."""
    inside_mask = _coherent_legacy_mask(image, detections, padding=16)
    outside_mask = _coherent_legacy_mask(image, detections, padding=24)
    inside = runner.run(image, Image.fromarray(inside_mask))
    outside = runner.run(image, Image.fromarray(outside_mask))
    legacy = [d for d in detections if d.name == "baozimh_top_left"]
    result = (
        _compose_clipped_bubble_repair(inside, outside, legacy[0])
        if legacy
        else outside
    )
    return result, outside_mask


def _detailed_logo_refinement_masks(
    image: Image.Image,
    detections: list[Detection],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build optional icon and dark-detail masks for a second Anime-LaMa pass."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, icon_mask, _ = _prepare_dense_overlap_hybrid(image, detections)

    icon_polish: np.ndarray | None = None
    if np.any(icon_mask):
        icon_polish = cv2.dilate(
            icon_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        core = np.zeros(gray.shape, dtype=bool)
        for detection in detections:
            if detection.name == "baozimh_logo":
                core |= detection.mask > 0
        ring = (icon_polish > 0) & ~core
        # Dense speech lettering around the icon is more valuable than a polish
        # pass and should remain protected by the first reconstruction.
        if not np.any(ring) or float(np.mean(gray[ring] < 100)) >= 0.24:
            icon_polish = None

    dark_detail: np.ndarray | None = None
    if _dense_logo_spans_dark_artwork(image, detections):
        precise = combine_cleanup_masks(image, detections, logo_style="precise")
        dilated = cv2.dilate(
            precise,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        ) > 0
        closed_dark = cv2.morphologyEx(
            (gray < 155).astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        ) > 0
        candidate = (dilated & closed_dark).astype(np.uint8) * 255
        if np.any(candidate):
            dark_detail = candidate
    return icon_polish, dark_detail


def _legacy_banner_cleanup_mask(
    image: Image.Image,
    detections: list[Detection],
) -> np.ndarray:
    """Build a solid cleanup mask for the old translucent top-left banner.

    The legacy ``最新免费漫画 / baozimh.com`` watermark is not just text and
    arrows: it also contains a translucent rounded white panel.  A foreground-
    shaped mask removes the lettering but leaves that washed rectangle behind.
    For this legacy variant, inpaint the aligned banner box as one region while
    punching genuine very-dark comic structure back out unless it belongs to the
    detected watermark foreground itself.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out = np.zeros(gray.shape, dtype=np.uint8)

    for det in detections:
        if det.name != "baozimh_top_left":
            continue
        x, y, w, h = det.box
        if w <= 0 or h <= 0:
            continue
        x2 = min(image.width, x + w)
        y2 = min(image.height, y + h)
        if x >= x2 or y >= y2:
            continue

        local = np.full((y2 - y, x2 - x), 255, dtype=np.uint8)
        core = det.mask[y:y2, x:x2] > 0
        dark = ((gray[y:y2, x:x2] < 92) & ~core).astype(np.uint8)
        protected = cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=1) > 0
        protected &= ~core
        local[protected] = 0
        out[y:y2, x:x2] = np.maximum(out[y:y2, x:x2], local)

    return out


def _fill_promo_blocks(
    image: Image.Image,
    detections: list[Detection],
) -> tuple[Image.Image, np.ndarray]:
    """Remove validated opaque Baozimh promo strips with their own background.

    These strips are deliberately detected only on nearly-white page-boundary
    panels.  A deterministic fill is both cleaner and safer than asking an
    inpainting model to hallucinate across the full 850-pixel-wide attribution
    block.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for detection in detections:
        if detection.name != "baozimh_promo":
            continue
        x, y, w, h = detection.box
        x2 = min(rgb.shape[1], x + w)
        y2 = min(rgb.shape[0], y + h)
        if x >= x2 or y >= y2:
            continue
        roi = rgb[y:y2, x:x2]
        channel_spread = roi.max(axis=2) - roi.min(axis=2)
        background = roi[(roi.min(axis=2) >= 230) & (channel_spread <= 18)]
        color = (
            np.median(background, axis=0).astype(np.uint8)
            if len(background)
            else np.array([255, 255, 255], dtype=np.uint8)
        )
        rgb[y:y2, x:x2] = color
        mask = cv2.bitwise_or(mask, detection.mask)
    return Image.fromarray(rgb), mask


def _inside_promo_strip(detection: Detection, promo_mask: np.ndarray) -> bool:
    if detection.name == "baozimh_promo":
        return False
    x, y, w, h = detection.box
    if w <= 0 or h <= 0:
        return False
    x2 = min(promo_mask.shape[1], x + w)
    y2 = min(promo_mask.shape[0], y + h)
    if x >= x2 or y >= y2:
        return False
    return float(np.mean(promo_mask[y:y2, x:x2] > 0)) >= 0.95


def _apply_surface_plan(
    image: Image.Image,
    plan: OverlayPlan,
    prediction: np.ndarray,
    *,
    compatibility_full_box: bool = False,
) -> tuple[Image.Image, np.ndarray]:
    """Apply only a validated surface to the plan's missing pixels.

    ``compatibility_full_box`` is used by the legacy helper functions that
    historically reported a complete aligned box.  Even there, connected
    trusted structure is never overwritten; the planner's normal runtime path
    uses its narrower support mask.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    operation = np.zeros(rgb.shape[:2], dtype=np.uint8)
    if not plan.surface.valid or prediction.shape != rgb.shape:
        return Image.fromarray(rgb), operation

    apply = plan.deterministic_mask > 0
    if compatibility_full_box:
        x, y, w, h = plan.box
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(rgb.shape[1], x + max(0, w))
        y1 = min(rgb.shape[0], y + max(0, h))
        apply = np.zeros(rgb.shape[:2], dtype=bool)
        if x0 < x1 and y0 < y1:
            apply[y0:y1, x0:x1] = True
        apply &= plan.trusted_source_mask == 0

    rgb[apply] = prediction[apply]
    operation[apply] = 255
    return Image.fromarray(rgb), operation


def _preserve_trusted_sources(
    result: Image.Image,
    source: Image.Image,
    plans: list[OverlayPlan] | tuple[OverlayPlan, ...],
) -> Image.Image:
    """Restore trusted source pixels after every model/refinement pass."""
    if not plans:
        return result
    rgb = np.asarray(result.convert("RGB"), dtype=np.uint8).copy()
    source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    trusted = np.zeros(rgb.shape[:2], dtype=bool)
    for plan in plans:
        if plan.trusted_source_mask.shape == trusted.shape:
            trusted |= plan.trusted_source_mask > 0
    rgb[trusted] = source_rgb[trusted]
    return Image.fromarray(rgb)


def _apply_local_residual_repair(
    image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """Repair an observed residual with a single narrow deterministic pass.

    Residual correction is intentionally kept out of the generative runner.
    The candidate mask has already been registered and intersected with the
    prior authorized support, so Telea only fills the observed fringe.  The
    explicit composite protects every unmasked pixel from backend-context
    changes and makes operation-mask provenance exact.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    local_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if local_mask.shape != rgb.shape[:2] or not np.any(local_mask):
        return image.copy()
    repaired_bgr = cv2.inpaint(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        local_mask,
        2,
        cv2.INPAINT_TELEA,
    )
    repaired = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB)
    repaired[local_mask == 0] = rgb[local_mask == 0]
    return Image.fromarray(repaired)


def _union_plan_masks(
    plans: list[OverlayPlan] | tuple[OverlayPlan, ...],
    shape: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    for plan in plans:
        mask = plan.missing_information_mask
        if mask.shape == shape:
            result = cv2.bitwise_or(result, mask)
    return result


def _append_qa_reasons(
    plan: OverlayPlan,
    *reasons: str,
) -> None:
    existing = list(plan.qa_reasons)
    for reason in reasons:
        if reason and reason not in existing:
            existing.append(reason)
    plan.qa_reasons = tuple(existing)


def _set_residual_evidence(
    plan: OverlayPlan,
    residual: ResidualCheck,
) -> None:
    """Copy one residual decision into the per-overlay plan metadata."""

    plan.residual = ResidualEvidence(
        registered_components=len(residual.components),
        candidate_pixels=int(np.count_nonzero(residual.candidate_mask)),
        qualifying_components=sum(
            1
            for component in residual.components
            if bool(component.get("qualifies"))
        ),
        max_coverage=max(
            (float(component.get("coverage", 0.0)) for component in residual.components),
            default=0.0,
        ),
        should_flag=residual.should_flag,
        reason=(
            "residual_watermark_evidence"
            if residual.should_flag
            else None
        ),
    )


def _legacy_registration_supports_full_banner_cleanup(
    plan: OverlayPlan,
) -> bool:
    """Return whether a weak panel fit still has exact banner registration.

    The legacy detector's aligned 325x92 registration identifies the complete
    rounded banner even when its translucent panel cannot be inverse-fitted
    against a structured or nearly white scene.  A narrow foreground-only
    fallback is incorrect for that case because the registered overlay also
    contains a panel and anti-aliased URL suffix.  Broad banner cleanup remains
    gated on independent high-confidence identity and registration evidence;
    uncertain detections keep the localized fallback.
    """
    return bool(
        plan.family == "legacy"
        and plan.identity.variant == "legacy-banner"
        and plan.identity.confidence >= 0.90
        and plan.registration.confidence >= 0.90
        and plan.registration.source == "detector-evidence"
        and plan.registration.offset == (0, 0)
        and "aligned-detection-box" in plan.registration.evidence
    )


class Cleaner:
    """Reusable watermark-cleaning engine shared by ``clean`` and pipelines."""

    def __init__(self, settings: CleanSettings) -> None:
        self.settings = settings
        self.inpainter = Inpainter(settings.backend, settings.device)
        self.comic_inpainter: Inpainter | None = None

        if settings.comic_overlap_backend == "anime-lama" or (
            settings.comic_overlap_backend == "auto" and self.inpainter.backend == "lama"
        ):
            self.comic_inpainter = Inpainter("anime-lama", settings.device)

    @property
    def primary_backend(self) -> str:
        return self.inpainter.backend

    def clean(self, image: Image.Image) -> CleanImageResult:
        image = image.convert("RGB")
        settings = self.settings
        detections = detect_watermarks(
            image,
            aggressive=settings.aggressive,
            dilation=settings.dilation,
            min_confidence=settings.min_confidence,
        )
        # Build independent evidence before any destructive operation.  The
        # surface predictions are kept separate from their plans so metadata
        # can expose the decision without serializing full image arrays.
        plans, surface_predictions = build_overlay_plans(
            image,
            detections,
            legacy_banner_mode=settings.legacy_banner_mode,
        )
        if not detections:
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
            return CleanImageResult(
                image=image.copy(),
                mask=mask,
                detections=detections,
                cleaned=False,
                backend=None,
                overlap_score=0.0,
                plans=tuple(plans),
                operation_mask=mask,
                provenance={"planner": "recovery-plan-v1", "plans": []},
            )

        promo_detections = [d for d in detections if d.name == "baozimh_promo"]
        promo_result, promo_mask = _fill_promo_blocks(image, promo_detections)
        result = promo_result
        operation_mask = promo_mask.copy()
        plan_by_detection = {
            id(detection): plan
            for detection, plan in zip(detections, plans)
        }

        # Structural overlap is evaluated per detection.  The aggregate value
        # remains for the compatibility field, but it no longer authorizes a
        # different detection's repair route.
        non_promo_detections = [
            d
            for d in detections
            if d.name != "baozimh_promo" and not _inside_promo_strip(d, promo_mask)
        ]
        overlap_score = max_structural_overlap_score(image, non_promo_detections)
        per_detection_overlap: dict[int, float] = {}
        for detection in non_promo_detections:
            per_detection_overlap[id(detection)] = max_structural_overlap_score(
                image,
                [detection],
            )

        # Dense modern repair remains an explicit successful compatibility
        # route, but is selected independently for each logo.  Moderate overlap
        # also keeps a logo out of deterministic surface filling.
        route_overrides: dict[int, str] = {}
        for detection in non_promo_detections:
            plan = plan_by_detection[id(detection)]
            if plan.route == "qa-hold":
                continue
            if detection.name != "baozimh_logo":
                continue
            overlap = per_detection_overlap.get(id(detection), 0.0)
            if (
                settings.logo_mask_style == "solid"
                and overlap >= settings.comic_dense_overlap_threshold
            ):
                route_overrides[id(detection)] = (
                    "modern-dense-hybrid"
                    if _dense_logo_spans_dark_artwork(image, [detection])
                    else "modern-dense-precise"
                )
                plan.route = route_overrides[id(detection)]
            elif (
                getattr(getattr(detection, "evidence", None), "glyph_mode", None)
                == "occlusion-consistent"
                and _dense_logo_spans_dark_artwork(image, [detection])
            ):
                # A validated occluded logo over broad dark artwork has the
                # same failure mode as the dense-skirt guard: a precise
                # generative pass copies translucent glyphs from its context.
                # Keep generation to the opaque icon and use deterministic
                # local repair for the wordmark.
                route_overrides[id(detection)] = "modern-dense-hybrid"
                plan.route = "modern-dense-hybrid"
            elif overlap >= settings.comic_overlap_threshold and plan.route.startswith(
                "surface-"
            ):
                route_overrides[id(detection)] = "modern-localized"
                plan.route = "modern-localized"
                _append_qa_reasons(plan, "surface_fit_rejected")

        # Apply only plans validated as deterministic surfaces.  This replaces
        # the old page-wide pale predicates and leaves connected structure
        # untouched by construction.
        surface_plans: list[OverlayPlan] = []
        for detection, plan in zip(detections, plans):
            route = route_overrides.get(id(detection), plan.route)
            if route not in {"surface-constant", "surface-affine"}:
                continue
            if detection.name == "baozimh_promo":
                continue
            prediction = surface_predictions.get(plan.overlay_id)
            if prediction is None:
                continue
            result, applied = _apply_surface_plan(result, plan, prediction)
            operation_mask = cv2.bitwise_or(operation_mask, applied)
            surface_plans.append(plan)

        active_pairs = [
            (detection, plan)
            for detection, plan in zip(detections, plans)
            if detection.name != "baozimh_promo"
            and not _inside_promo_strip(detection, promo_mask)
            and plan not in surface_plans
            and plan.route != "qa-hold"
        ]
        active_detections = [detection for detection, _ in active_pairs]
        used_dense_logo_mask = any(
            route_overrides.get(id(detection), plan.route)
            in {"modern-dense-hybrid", "modern-dense-precise"}
            for detection, plan in active_pairs
        )
        dense_repair_mode: str | None = None
        if used_dense_logo_mask:
            dense_repair_mode = (
                "hybrid"
                if any(
                    route_overrides.get(id(detection), plan.route)
                    == "modern-dense-hybrid"
                    for detection, plan in active_pairs
                )
                else "precise"
            )
        used_legacy_banner_mask = any(
            plan.family == "legacy" for plan in plans
        )

        # A model runner is chosen for each active plan.  This is intentionally
        # local so one dense/detail page cannot force a neighboring detection
        # into the same broad route.
        used_comic_fallback = False
        model_used = False
        deterministic_backend_used = False
        legacy_repairs = []
        residual_checks: list[ResidualCheck] = []
        residual_operation = np.zeros((image.height, image.width), dtype=np.uint8)

        for detection, plan in active_pairs:
            route = route_overrides.get(id(detection), plan.route)
            if self.comic_inpainter is not None and (
                plan.family == "legacy"
                or route.startswith("modern-")
                or per_detection_overlap.get(id(detection), 0.0)
                >= settings.comic_overlap_threshold
            ):
                runner = self.comic_inpainter
                used_comic_fallback = runner is not self.inpainter
            else:
                runner = self.inpainter

            if plan.family == "legacy":
                # Explicit modes remain available.  In auto mode, constrained
                # geometry uses the precise path; otherwise the established
                # panel fit is attempted and rejected if weak.
                if route == "legacy-precise":
                    legacy_mode = "precise"
                elif route == "legacy-coherent":
                    legacy_mode = "coherent"
                elif route == "legacy-solid":
                    legacy_mode = "solid"
                elif route == "legacy-deblend":
                    legacy_mode = "deblend"
                else:
                    legacy_mode = "auto"

                clipped_bubble = (
                    legacy_mode == "auto"
                    and plan.scene.geometry_kind == "circle"
                    and plan.scene.geometry_supported
                    and _legacy_banner_overlaps_clipped_bubble(image, [detection])
                )
                if clipped_bubble:
                    repair = prepare_legacy_banner_repair(
                        result,
                        [detection],
                        mode="coherent",
                    )
                    result, bubble_mask = _repair_clipped_bubble_banner(
                        result,
                        [detection],
                        runner,
                    )
                    repair.mode_used = "bubble-coherent"
                    repair.ai_mask = bubble_mask
                    repair.report_mask = bubble_mask
                    repair.ai_mask_pixel_count = int(np.count_nonzero(bubble_mask))
                    repair.ai_mask_fraction = (
                        repair.ai_mask_pixel_count / repair.panel_pixel_count
                        if repair.panel_pixel_count
                        else 0.0
                    )
                    operation_mask = cv2.bitwise_or(operation_mask, bubble_mask)
                    model_used = True
                else:
                    repair = prepare_legacy_banner_repair(
                        result,
                        [detection],
                        mode=legacy_mode,
                    )
                    localized_legacy_fallback = False
                    # A weak/failed fit is evidence for QA.  A strongly aligned
                    # complete banner can still use the established registered
                    # panel path; uncertain registration stays on the precise
                    # mask so a fit failure cannot authorize broad cleanup.
                    weak_fit = repair.fit is None or repair.fit.improvement < 0.20
                    calibrated_banner = _legacy_registration_supports_full_banner_cleanup(
                        plan
                    )
                    if (
                        legacy_mode == "auto"
                        and weak_fit
                        and plan.scene.structure_pixels > 0
                        and not calibrated_banner
                    ):
                        _append_qa_reasons(plan, "weak_overlay_fit")
                        _append_qa_reasons(plan, "unstable_recovery")
                        repair = prepare_legacy_banner_repair(
                            result,
                            [detection],
                            mode="precise",
                        )
                        _append_qa_reasons(plan, "localized_legacy_fallback")
                        localized_legacy_fallback = True
                    elif legacy_mode == "auto" and weak_fit and calibrated_banner:
                        # The registered banner is complete even though the
                        # current recovery model is not validated.  Keep auto's
                        # calibrated full-banner + seam path: the old precise
                        # override omitted the panel and about 1.6k foreground
                        # support pixels, leaving readable title/URL fragments.
                        plan.routing = plan.routing.__class__(
                            route="legacy-calibrated-banner",
                            generation_scope="broad-compatible",
                            reason="recovery-model-unvalidated",
                            compatibility_mode="auto",
                        )
                        plan.provenance["legacy_calibrated_banner"] = {
                            "identity_confidence": float(plan.identity.confidence),
                            "registration_confidence": float(
                                plan.registration.confidence
                            ),
                            "registration_source": plan.registration.source,
                            "fit_improvement": (
                                float(repair.fit.improvement)
                                if repair.fit is not None
                                else None
                            ),
                            "foreground_only_fallback_rejected": True,
                        }
                        _append_qa_reasons(plan, "weak_overlay_fit")
                        _append_qa_reasons(plan, "recovery_model_unvalidated")
                        _append_qa_reasons(plan, "broad_legacy_generation")
                    if localized_legacy_fallback:
                        source_array = np.asarray(
                            repair.image.convert("RGB"), dtype=np.uint8
                        )
                        telea = cv2.inpaint(
                            cv2.cvtColor(source_array, cv2.COLOR_RGB2BGR),
                            repair.ai_mask,
                            5,
                            cv2.INPAINT_TELEA,
                        )
                        generated = Image.fromarray(
                            cv2.cvtColor(telea, cv2.COLOR_BGR2RGB)
                        )
                        repair.deterministic_mask = repair.ai_mask.copy()
                        repair.deterministic_pixel_count = int(
                            np.count_nonzero(repair.ai_mask)
                        )
                        if repair.missing_mask is not None:
                            repair.missing_mask[
                                repair.deterministic_mask > 0
                            ] = 0
                            repair.missing_pixel_count = int(
                                np.count_nonzero(repair.missing_mask)
                            )
                        repair.generated_mask = np.zeros_like(repair.ai_mask)
                        repair.generated_pixel_count = 0
                        repair.ai_mask_pixel_count = 0
                        repair.ai_mask_fraction = 0.0
                        deterministic_backend_used = True
                    else:
                        generated = runner.run(
                            repair.image,
                            Image.fromarray(repair.ai_mask),
                        )
                    result = compose_legacy_banner_repair(
                        result,
                        generated,
                        repair,
                    )
                    operation_mask = cv2.bitwise_or(
                        operation_mask,
                        repair.report_mask,
                    )
                    model_used = model_used or not localized_legacy_fallback

                    if (
                        legacy_mode == "auto"
                        and repair.mode_used in {"solid", "solid-polished"}
                    ):
                        # Auto's compatibility route reconstructs most of the
                        # registered banner.  Preserve the result but expose
                        # that broad-generation fact for downstream/manual QA;
                        # a good boundary fit alone cannot validate hidden
                        # source structure in the panel interior.
                        plan.routing = plan.routing.__class__(
                            route="legacy-broad-compatible",
                            generation_scope="broad-compatible",
                            reason="recovery-model-unvalidated",
                            compatibility_mode="auto",
                        )
                        _append_qa_reasons(plan, "recovery_model_unvalidated")
                        _append_qa_reasons(plan, "broad_legacy_generation")
                    elif repair.mode_used == "broad-fallback":
                        _append_qa_reasons(plan, "recovery_model_unvalidated")
                        _append_qa_reasons(plan, "broad_legacy_generation")

                    if (
                        repair.mode_used in {"solid", "solid-polished"}
                        and repair.seam_mask is not None
                        and np.any(repair.seam_mask)
                    ):
                        pre_seam = np.asarray(result, dtype=np.uint8).copy()
                        seam_result = runner.run(
                            result,
                            Image.fromarray(repair.seam_mask),
                        )
                        if repair.seam_blend_mask is None:
                            blended = np.asarray(seam_result, dtype=np.uint8)
                        else:
                            alpha = repair.seam_blend_mask[..., None]
                            blended = np.clip(
                                pre_seam.astype(np.float32) * (1.0 - alpha)
                                + np.asarray(seam_result, dtype=np.float32) * alpha,
                                0,
                                255,
                            ).astype(np.uint8)
                        # Backends may return small changes outside their
                        # requested seam mask.  Keep the compatibility result
                        # identical to the unseamed repair outside the explicit
                        # boundary/seam blend support.
                        seam_support = np.zeros(operation_mask.shape, dtype=bool)
                        if repair.boundary_blend_mask is not None:
                            seam_support |= repair.boundary_blend_mask > 0
                        if repair.seam_blend_mask is not None:
                            seam_support |= repair.seam_blend_mask > 0
                        blended[~seam_support] = pre_seam[~seam_support]
                        result = Image.fromarray(blended)
                        seam_disposition = seam_support.astype(np.uint8) * 255
                        repair.generated_mask = cv2.bitwise_or(
                            repair.generated_mask,
                            seam_disposition,
                        )
                        for disposition_name in (
                            "validated_recovered_mask",
                            "deterministic_mask",
                            "trusted_mask",
                        ):
                            disposition = getattr(repair, disposition_name)
                            if disposition is not None:
                                disposition[seam_support] = 0
                        if repair.authorized_support_mask is not None:
                            support = repair.authorized_support_mask > 0
                            failed = (
                                repair.failed_mask > 0
                                if repair.failed_mask is not None
                                else np.zeros_like(support)
                            )
                            repair.missing_mask = (
                                ((repair.generated_mask > 0) & support) | failed
                            ).astype(np.uint8) * 255
                        repair.generated_pixel_count = int(
                            np.count_nonzero(repair.generated_mask)
                        )
                        repair.validated_recovered_pixel_count = int(
                            np.count_nonzero(repair.validated_recovered_mask)
                        )
                        repair.deterministic_pixel_count = int(
                            np.count_nonzero(repair.deterministic_mask)
                        )
                        repair.trusted_pixel_count = int(
                            np.count_nonzero(repair.trusted_mask)
                        )
                        repair.missing_pixel_count = int(
                            np.count_nonzero(repair.missing_mask)
                        )
                        repair.recovery_promoted = bool(
                            repair.validated_recovered_pixel_count
                        )
                        operation_mask = cv2.bitwise_or(
                            operation_mask,
                            (seam_support.astype(np.uint8) * 255),
                        )
                        model_used = True
                # ``prepare_legacy_banner_repair`` already partitions strong
                # inverse fits into recovered/source and generated regions.
                # Do not restore the broad heuristic structure mask over a
                # validated deblend/auto result: that mask can include the
                # translucent foreground itself.  Precise/localized fallbacks
                # still get the strict source-preservation guard.
                preserve_legacy = (
                    repair.mode_used in {"precise", "coherent", "solid", "bubble-coherent"}
                    or "localized_legacy_fallback" in plan.qa_reasons
                )
                if preserve_legacy:
                    result = _preserve_trusted_sources(result, image, [plan])
                if repair.trusted_mask is not None:
                    plan.trusted_source_mask = repair.trusted_mask.copy()
                if repair.validated_recovered_mask is not None:
                    plan.recovered_mask = repair.validated_recovered_mask.copy()
                if repair.deterministic_mask is not None:
                    plan.deterministic_mask = repair.deterministic_mask.copy()
                if repair.missing_mask is not None:
                    plan.missing_information_mask = repair.missing_mask.copy()
                if repair.generated_mask is not None:
                    plan.generated_mask = repair.generated_mask.copy()
                if repair.failed_mask is not None:
                    plan.failed_mask = repair.failed_mask.copy()
                broad_generation = (
                    plan.routing.generation_scope == "broad-compatible"
                    or repair.mode_used
                    in {
                        "broad-fallback",
                        "coherent",
                        "solid",
                        "solid-polished",
                        "bubble-coherent",
                    }
                    or "broad_legacy_generation" in plan.qa_reasons
                )
                recovery_evidence: list[str] = []
                if repair.validated_recovered_pixel_count:
                    recovery_evidence.append("validated_legacy_inverse_recovery")
                if repair.deterministic_pixel_count:
                    recovery_evidence.append("executed_deterministic_reconstruction")
                if repair.ai_mask_pixel_count:
                    recovery_evidence.append(
                        "broad_compatibility_reconstruction"
                        if broad_generation
                        else "localized_reconstruction"
                    )
                if repair.failed_pixel_count:
                    recovery_evidence.append("explicit_qa_failure")
                plan.recovery = plan.recovery.__class__(
                    confidence=(
                        float(repair.stability.stable_fraction)
                        if repair.stability is not None
                        else 0.0
                    ),
                    recoverable_pixels=repair.validated_recovered_pixel_count,
                    deterministic_pixels=repair.deterministic_pixel_count,
                    localized_pixels=repair.missing_pixel_count,
                    broad_pixels=(
                        repair.ai_mask_pixel_count if broad_generation else 0
                    ),
                    fit_valid=bool(
                        repair.fit
                        and repair.fit.promotion_eligible
                        and repair.validated_recovered_pixel_count
                    ),
                    fit_stability=(
                        float(repair.stability.stable_fraction)
                        if repair.stability is not None
                        else 0.0
                    ),
                    evidence=tuple(recovery_evidence),
                )
                if repair.fit is not None and repair.fit.promotion_eligible:
                    # Scene-only warnings are provisional until the overlay fit
                    # proves that the accepted compatibility route is stable.
                    plan.qa_reasons = tuple(
                        reason
                        for reason in plan.qa_reasons
                        if reason
                        not in {
                            "weak_overlay_fit",
                            "unstable_recovery",
                            "structural_discontinuity",
                            "unsupported_geometry_reconstruction",
                        }
                    )
                for reason in repair.qa_reasons:
                    _append_qa_reasons(plan, reason)
                plan.provenance["legacy_repair"] = {
                    "panel_pixels": repair.panel_pixel_count,
                    "candidate_recovered_pixels": repair.candidate_recovered_pixel_count,
                    "validated_recovered_pixels": repair.validated_recovered_pixel_count,
                    "deterministic_pixels": repair.deterministic_pixel_count,
                    "missing_pixels": repair.missing_pixel_count,
                    "generated_pixels": repair.generated_pixel_count,
                    "reconstruction_operation_pixels": repair.ai_mask_pixel_count,
                    "failed_pixels": repair.failed_pixel_count,
                    "numeric_fit_eligible": bool(
                        repair.fit and repair.fit.promotion_eligible
                    ),
                    "semantic_model_validated": False,
                    "partition_complete": bool(
                        repair.authorized_support_mask is not None
                        and np.all(
                            np.sum(
                                np.stack(
                                    [
                                        plan.trusted_source_mask > 0,
                                        plan.recovered_mask > 0,
                                        plan.deterministic_mask > 0,
                                        plan.generated_mask > 0,
                                        plan.failed_mask > 0,
                                    ]
                                )[:, repair.authorized_support_mask > 0],
                                axis=0,
                            )
                            == 1
                        )
                    ),
                    "recovery_promoted": repair.recovery_promoted,
                }
                legacy_repairs.append(repair)
                continue

            # Modern routes always receive a mask derived from missing
            # information.  The detector helper remains the source of the
            # compatibility shape, but its result is intersected with the
            # per-overlay plan before model execution.
            dense = route in {"modern-dense-hybrid", "modern-dense-precise"}
            precise = route in {"modern-dense-precise", "modern-localized"}
            compatibility_mask = combine_cleanup_masks(
                result,
                [detection],
                logo_style="precise" if precise else settings.logo_mask_style,
                padding=(
                    max(6, settings.logo_mask_padding)
                    if precise
                    else settings.logo_mask_padding
                ),
                protect_dark=True,
            )
            safe_mask = cv2.bitwise_and(
                compatibility_mask,
                plan.missing_information_mask,
            )
            if route == "modern-dense-hybrid":
                dense_input, dense_ai_mask, dense_report_mask = (
                    _prepare_dense_overlap_hybrid(result, [detection])
                )
                result = runner.run(
                    dense_input,
                    Image.fromarray(dense_ai_mask),
                )
                safe_mask = dense_report_mask
                plan.generated_mask = cv2.bitwise_or(
                    plan.generated_mask,
                    dense_ai_mask,
                )
                dense_deterministic = cv2.bitwise_and(
                    dense_report_mask,
                    cv2.bitwise_not(dense_ai_mask),
                )
                plan.deterministic_mask = cv2.bitwise_or(
                    plan.deterministic_mask,
                    dense_deterministic,
                )
            else:
                result = runner.run(result, Image.fromarray(safe_mask))
                plan.generated_mask = cv2.bitwise_or(
                    plan.generated_mask,
                    safe_mask,
                )
            model_used = True
            operation_mask = cv2.bitwise_or(operation_mask, safe_mask)
            result = _preserve_trusted_sources(result, image, [plan])

            # Refinement is allowed only for an unstructured/detailed modern
            # route.  Every refinement mask is part of the reported operation
            # union so QA can reason about the full changed area.
            if (
                not dense
                and plan.scene.scene_class not in {
                    "structured-surface",
                    "bubble-geometry",
                    "card-geometry",
                }
                and runner.backend == "anime-lama"
                and route != "modern-localized"
            ):
                icon_polish, dark_detail = _detailed_logo_refinement_masks(
                    image,
                    [detection],
                )
                for refinement in (icon_polish, dark_detail):
                    if refinement is None:
                        continue
                    refinement = cv2.bitwise_and(
                        refinement,
                        plan.missing_information_mask,
                    )
                    if not np.any(refinement):
                        continue
                    result = runner.run(result, Image.fromarray(refinement))
                    result = _preserve_trusted_sources(result, image, [plan])
                    operation_mask = cv2.bitwise_or(operation_mask, refinement)
                    model_used = True

            # One residual gate applies to every modern route, including the
            # dense hybrid.  It returns narrow candidate evidence for QA, but
            # it does not authorize a second cleanup pass.  In particular,
            # Telea on a registered-looking component can smear source
            # structure (the observed 9/11 and 11/39 class), so candidate
            # pixels remain diagnostic until a future independently validated
            # repair route exists.  The candidate itself is still intersected
            # with this plan's already-authorized operation and
            # missing-information support.
            authorized_residual = cv2.bitwise_and(
                plan.missing_information_mask,
                operation_mask,
            )
            residual = evaluate_registered_residual(
                result,
                [detection],
                source_image=image,
                authorized_mask=authorized_residual,
                artwork_veto_mask=plan.trusted_source_mask,
            )
            residual_checks.append(residual)
            _set_residual_evidence(plan, residual)
            if residual.should_flag:
                # Residual evidence remains an acceptance/QA signal even when
                # the stricter correction gates reject it.  No broad retry is
                # permitted; notably this keeps the dense-dark 3/0027 guard
                # unchanged.
                _append_qa_reasons(plan, "residual_watermark_evidence")

            if np.any(residual.operation_mask):
                # Keep this as evidence only.  The candidate is useful for
                # reporting the exact registered support that still appears
                # contaminated, but promoting it to an automatic inpaint
                # would make the cleaner overwrite source artwork when the
                # residual metric is a false positive.  This is deliberately
                # generic: no chapter/page-specific exception is needed for
                # the known smudging class or for the weaker 9/25 class.
                _append_qa_reasons(plan, "residual_correction_deferred")
                plan.provenance["residual_correction"] = {
                    "status": "deferred-qa",
                    "candidate_pixels": int(np.count_nonzero(residual.operation_mask)),
                    "source_palette_gated": True,
                    "registered_template_gated": True,
                    "authorized_support_intersection": True,
                    "artwork_veto_applied": True,
                    "automatic_repair": False,
                }

        # The promotion strip is deterministic and must remain the last write
        # if a neighbouring model used a padded context.  Trusted source and
        # operation-mask invariants are enforced before exposing the result.
        if promo_detections:
            result, promo_final_mask = _fill_promo_blocks(result, promo_detections)
            operation_mask = cv2.bitwise_or(operation_mask, promo_final_mask)
        final_preservation_plans = [
            plan
            for plan in plans
            if plan.family != "legacy"
            or plan.route in {"legacy-precise", "legacy-coherent", "legacy-solid"}
            or "localized_legacy_fallback" in plan.qa_reasons
        ]
        result = _preserve_trusted_sources(result, image, final_preservation_plans)
        source_rgb = np.asarray(image, dtype=np.uint8)
        result_rgb = np.asarray(result.convert("RGB"), dtype=np.uint8).copy()
        changed = np.any(result_rgb != source_rgb, axis=2)
        outside_operation = changed & (operation_mask == 0)
        if np.any(outside_operation):
            # Some model backends use padded contexts.  Keep the contract that
            # the returned operation mask describes every permitted change.  A
            # backend-context expansion is recorded explicitly; silently
            # restoring those pixels would make the compatibility auto route
            # diverge from its validated seam pass.
            operation_mask[outside_operation] = 255
            for plan in plans:
                _append_qa_reasons(plan, "incomplete_operation_provenance")

        trusted = np.zeros((image.height, image.width), dtype=bool)
        for plan in final_preservation_plans:
            if plan.trusted_source_mask.shape == trusted.shape:
                trusted |= plan.trusted_source_mask > 0
        if np.any(np.any(np.asarray(result) != source_rgb, axis=2) & trusted):
            for plan in plans:
                _append_qa_reasons(plan, "trusted_source_overwritten")

        qa_reasons: list[str] = []
        for plan in plans:
            for reason in plan.qa_reasons:
                if reason == "localized_legacy_fallback":
                    continue
                if reason not in qa_reasons:
                    qa_reasons.append(reason)
        for residual in residual_checks:
            if residual.should_flag:
                if "residual_watermark_evidence" not in qa_reasons:
                    qa_reasons.append("residual_watermark_evidence")

        # Compatibility metrics retain the historical first/aggregate legacy
        # values while the plans/provenance contain per-overlay accounting.
        first_repair = legacy_repairs[0] if legacy_repairs else None
        total_panel = sum(int(r.panel_pixel_count) for r in legacy_repairs)
        total_deblended = sum(int(r.deblended_pixel_count) for r in legacy_repairs)
        total_ai = sum(int(r.ai_mask_pixel_count) for r in legacy_repairs)
        total_structural = sum(int(r.structural_pixel_count) for r in legacy_repairs)
        total_boundary = sum(int(r.boundary_pixel_count) for r in legacy_repairs)
        total_recovered_fraction = (
            sum(float(r.recovered_fraction) * int(r.panel_pixel_count) for r in legacy_repairs)
            / total_panel
            if total_panel
            else 0.0
        )
        total_deblended_fraction = total_deblended / total_panel if total_panel else 0.0
        total_ai_fraction = total_ai / total_panel if total_panel else 0.0
        total_boundary_fraction = total_boundary / total_panel if total_panel else 0.0
        legacy_mode_used = (
            "uniform-fill"
            if used_legacy_banner_mask and not legacy_repairs
            else (first_repair.mode_used if first_repair is not None else None)
        )
        fit = first_repair.fit if first_repair is not None else None
        backend = (
            (self.comic_inpainter.backend if used_comic_fallback else self.inpainter.backend)
            if model_used
            else ("opencv" if deterministic_backend_used else "solid-fill")
        )
        provenance = {
            "planner": "recovery-plan-v1",
            "mask_semantics": "union-of-deterministic-recovery-generation-polish-and-seam-operations",
            "operation_mask_pixels": int(np.count_nonzero(operation_mask)),
            "trusted_source_pixels": int(np.count_nonzero(trusted)),
            "residual_operation_pixels": int(np.count_nonzero(residual_operation)),
            "plans": [plan.metadata() for plan in plans],
            "residual_checks": [check.metadata() for check in residual_checks],
        }
        if not np.any(operation_mask):
            backend = None
        return CleanImageResult(
            image=result,
            mask=operation_mask,
            detections=detections,
            cleaned=bool(np.any(operation_mask)),
            backend=backend,
            overlap_score=overlap_score,
            used_comic_fallback=used_comic_fallback,
            used_dense_logo_mask=used_dense_logo_mask,
            dense_repair_mode=dense_repair_mode,
            used_legacy_banner_mask=used_legacy_banner_mask,
            legacy_banner_mode_used=legacy_mode_used,
            legacy_recovered_fraction=total_recovered_fraction,
            legacy_panel_pixel_count=total_panel,
            legacy_deblended_pixel_count=total_deblended,
            legacy_deblended_fraction=total_deblended_fraction,
            legacy_ai_mask_pixel_count=total_ai,
            legacy_ai_mask_fraction=total_ai_fraction,
            legacy_structural_pixel_count=total_structural,
            legacy_boundary_pixel_count=total_boundary,
            legacy_boundary_fraction=total_boundary_fraction,
            legacy_fit_space=(fit.working_space if fit else None),
            legacy_alpha=(fit.alpha if fit else None),
            legacy_fit_improvement=(fit.improvement if fit else 0.0),
            legacy_fallback_reason=(
                ",".join(
                    sorted(
                        {
                            str(r.fallback_reason)
                            for r in legacy_repairs
                            if r.fallback_reason
                        }
                    )
                )
                or None
            ),
            plans=tuple(plans),
            qa_reasons=tuple(qa_reasons),
            provenance=provenance,
            operation_mask=operation_mask,
            residual_mask=residual_operation,
        )
