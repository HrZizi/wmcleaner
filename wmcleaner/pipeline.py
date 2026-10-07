from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .cleaning import CleanImageResult, CleanSettings, Cleaner
from .scrapers import SourceDownloadResult, download_source


@dataclass(frozen=True)
class PipelinePage:
    source: str
    destination: str
    status: str
    backend: str | None
    overlap_score: float
    detections: tuple[str, ...]
    backup: str | None = None
    legacy_mode: str | None = None
    legacy_recovered_fraction: float = 0.0
    legacy_panel_pixels: int = 0
    legacy_deblended_pixels: int = 0
    legacy_deblended_fraction: float = 0.0
    legacy_ai_mask_pixels: int = 0
    legacy_ai_mask_fraction: float = 0.0
    legacy_structural_pixels: int = 0
    legacy_boundary_pixels: int = 0
    legacy_boundary_fraction: float = 0.0
    source_sha256: str | None = None
    output_sha256: str | None = None
    qa_reasons: tuple[str, ...] = ()
    cleaner_provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScrapeCleanResult:
    download: SourceDownloadResult
    scrape_root: Path
    clean_root: Path
    pages: tuple[PipelinePage, ...]
    cleaned_count: int
    unchanged_count: int
    backup_count: int
    in_place: bool


def default_clean_root(scrape_root: Path) -> Path:
    return scrape_root.with_name(scrape_root.name + "_cleaned")


def scrape_and_clean(
    source_url: str,
    output_dir: Path,
    *,
    chapter_selector: str = "all",
    delay: float = 1.0,
    overwrite: bool = False,
    write_manifest: bool = True,
    clean_settings: CleanSettings | None = None,
    clean_output: Path | None = None,
    in_place: bool = False,
    replaced_dir_name: str = "_replaced",
    save_masks: bool = False,
    debug: bool = False,
    jpeg_quality: int = 97,
) -> ScrapeCleanResult:
    """Download a chapter/series and immediately run watermark cleanup.

    In separate mode the scrape tree remains untouched and a complete mirrored
    reader-ready tree is written to ``clean_output`` (or ``<output>_cleaned``).
    In in-place mode only pages that were actually cleaned replace their scraped
    counterpart; each replaced original is moved into a per-chapter backup
    subfolder first.
    """
    output_dir = output_dir.expanduser().resolve()
    if clean_output is not None:
        clean_output = clean_output.expanduser().resolve()

    if in_place and clean_output is not None:
        raise ValueError("--clean-output cannot be combined with --in-place.")
    _validate_replaced_dir_name(replaced_dir_name)
    cleaner = Cleaner(clean_settings or CleanSettings())

    download = download_source(
        source_url,
        output_dir,
        chapter_selector=chapter_selector,
        delay=delay,
        overwrite=overwrite,
        write_manifest=write_manifest,
    )

    clean_root = output_dir if in_place else (clean_output or default_clean_root(output_dir))
    if not in_place and clean_root == output_dir:
        raise ValueError(
            "Separate cleaned output cannot equal the scrape directory. "
            "Use --in-place to replace scraped pages safely."
        )

    if not in_place:
        clean_root.mkdir(parents=True, exist_ok=True)
        if write_manifest:
            _copy_scrape_manifests(output_dir, clean_root, download.files)

    entries: list[PipelinePage] = []
    cleaned_count = 0
    unchanged_count = 0
    backup_count = 0

    for src in download.files:
        src = src.resolve()
        source_sha256 = _sha256_file(src)
        rel = src.relative_to(output_dir)
        dst = src if in_place else clean_root / rel
        image = Image.open(src).convert("RGB")
        result = cleaner.clean(image)
        detection_labels = tuple(
            f"{d.name}:{d.confidence:.2f}" for d in result.detections
        )

        if not result.cleaned:
            unchanged_count += 1
            qa = tuple(getattr(result, "qa_reasons", ()))
            held = bool(result.detections and qa)
            reason = f"QA hold: {', '.join(qa)}" if held else "no known watermark"
            if not in_place:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                status = "copied-qa" if held else "copied-unmatched"
                print(f"[copy]  {rel}: {reason} -> {dst}")
            else:
                status = "unchanged-qa" if held else "unchanged"
                print(f"[keep]  {rel}: {reason}")
            entries.append(
                PipelinePage(
                    source=rel.as_posix(),
                    destination=_relative_display(dst, clean_root),
                    status=status,
                    backend=None,
                    overlap_score=float(result.overlap_score),
                    detections=detection_labels,
                    source_sha256=source_sha256,
                    output_sha256=_sha256_file(dst),
                    qa_reasons=qa,
                    cleaner_provenance=dict(getattr(result, "provenance", {})),
                )
            )
            continue

        if result.used_comic_fallback:
            if getattr(result, "used_legacy_banner_mask", False):
                detail = getattr(result, "legacy_banner_mode_used", None) or "unknown"
                deblended = getattr(result, "legacy_deblended_fraction", 0.0)
                ai_fraction = getattr(result, "legacy_ai_mask_fraction", 0.0)
                structural = getattr(result, "legacy_structural_pixel_count", 0)
                boundary = getattr(result, "legacy_boundary_fraction", 0.0)
                fit_space = getattr(result, "legacy_fit_space", None)
                alpha = getattr(result, "legacy_alpha", None)
                fit = (
                    f" | {fit_space} alpha={alpha:.2f}" if fit_space and alpha is not None else ""
                )
                metrics = (
                    f"boundary-polish={boundary:.1%} ai={ai_fraction:.1%}"
                    if detail == "solid-polished"
                    else (
                        f"deblended={deblended:.1%} ai={ai_fraction:.1%} "
                        f"structure={structural}"
                    )
                )
                print(f"[legacy] {rel}: anime-lama {detail} {metrics}{fit}")
            else:
                tag = "comic-dense" if result.used_dense_logo_mask else "comic"
                suffix = (
                    f" | dense-{result.dense_repair_mode}-repair"
                    if result.dense_repair_mode
                    else ""
                )
                print(
                    f"[{tag}] {rel}: anime-lama "
                    f"structural-overlap={result.overlap_score:.3f}{suffix}"
                )

        backup: Path | None = None
        if in_place:
            backup_dir = src.parent / replaced_dir_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = _unique_backup_path(backup_dir / src.name)
            temp = _temporary_image_path(src)
            try:
                _save_image(result.image, temp, jpeg_quality=jpeg_quality)
                src.replace(backup)
                try:
                    temp.replace(src)
                except Exception:
                    backup.replace(src)
                    raise
            finally:
                temp.unlink(missing_ok=True)
            backup_count += 1
            print(f"[replace] {rel} | original -> {backup.relative_to(output_dir)}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            _save_image_atomic(result.image, dst, jpeg_quality=jpeg_quality)
            print(f"[clean] {rel} -> {dst}")

        if save_masks:
            mask_path = dst.with_name(dst.stem + ".mask.png")
            _save_mask(result.mask, mask_path)
        if debug:
            debug_path = dst.with_name(dst.stem + ".debug.png")
            _save_debug(image, result, debug_path)

        cleaned_count += 1
        entries.append(
            PipelinePage(
                source=rel.as_posix(),
                destination=_relative_display(dst, clean_root),
                status="replaced" if in_place else "cleaned",
                backend=result.backend,
                overlap_score=round(result.overlap_score, 6),
                detections=detection_labels,
                backup=(backup.relative_to(output_dir).as_posix() if backup else None),
                legacy_mode=getattr(result, "legacy_banner_mode_used", None),
                legacy_recovered_fraction=round(
                    float(getattr(result, "legacy_recovered_fraction", 0.0)), 6
                ),
                legacy_panel_pixels=int(
                    getattr(result, "legacy_panel_pixel_count", 0)
                ),
                legacy_deblended_pixels=int(
                    getattr(result, "legacy_deblended_pixel_count", 0)
                ),
                legacy_deblended_fraction=round(
                    float(getattr(result, "legacy_deblended_fraction", 0.0)), 6
                ),
                legacy_ai_mask_pixels=int(
                    getattr(result, "legacy_ai_mask_pixel_count", 0)
                ),
                legacy_ai_mask_fraction=round(
                    float(getattr(result, "legacy_ai_mask_fraction", 0.0)), 6
                ),
                legacy_structural_pixels=int(
                    getattr(result, "legacy_structural_pixel_count", 0)
                ),
                legacy_boundary_pixels=int(
                    getattr(result, "legacy_boundary_pixel_count", 0)
                ),
                legacy_boundary_fraction=round(
                    float(getattr(result, "legacy_boundary_fraction", 0.0)), 6
                ),
                source_sha256=source_sha256,
                output_sha256=_sha256_file(dst),
                qa_reasons=tuple(getattr(result, "qa_reasons", ())),
                cleaner_provenance=dict(getattr(result, "provenance", {})),
            )
        )

    pipeline_result = ScrapeCleanResult(
        download=download,
        scrape_root=output_dir,
        clean_root=clean_root,
        pages=tuple(entries),
        cleaned_count=cleaned_count,
        unchanged_count=unchanged_count,
        backup_count=backup_count,
        in_place=in_place,
    )
    if write_manifest:
        _write_pipeline_manifest(
            pipeline_result,
            source_url=source_url,
            chapter_selector=chapter_selector,
            clean_settings=clean_settings or CleanSettings(),
            replaced_dir_name=replaced_dir_name,
        )
    return pipeline_result


def _copy_scrape_manifests(
    scrape_root: Path,
    clean_root: Path,
    files: tuple[Path, ...],
) -> None:
    parents = {path.resolve().parent for path in files}
    for parent in parents:
        manifest = parent / "scrape-manifest.json"
        if manifest.is_file():
            rel = manifest.relative_to(scrape_root)
            dst = clean_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest, dst)

    series_manifest = scrape_root / "series-manifest.json"
    if series_manifest.is_file():
        shutil.copy2(series_manifest, clean_root / series_manifest.name)


def _write_pipeline_manifest(
    result: ScrapeCleanResult,
    *,
    source_url: str,
    chapter_selector: str,
    clean_settings: CleanSettings,
    replaced_dir_name: str,
) -> None:
    manifest = {
        "version": 1,
        "source_url": source_url,
        "source_kind": result.download.kind,
        "chapter_selector": chapter_selector,
        "layout": "in-place" if result.in_place else "separate",
        "replaced_dir": replaced_dir_name if result.in_place else None,
        "chapter_count": result.download.chapter_count,
        "page_count": len(result.pages),
        "cleaned_count": result.cleaned_count,
        "unchanged_count": result.unchanged_count,
        "backup_count": result.backup_count,
        "clean_settings": asdict(clean_settings),
        "pages": [asdict(page) for page in result.pages],
    }
    path = result.clean_root / "scrape-clean-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Pipeline manifest: {path}")


def _relative_display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_replaced_dir_name(value: str) -> None:
    invalid = '<>:"/\\|?*'
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or any(char in value for char in invalid)
        or value.endswith((" ", "."))
    ):
        raise ValueError(
            "replaced backup folder must be one Windows-safe directory name."
        )


def _unique_backup_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}.{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a backup filename for {path.name}")


def _temporary_image_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.wmclean.tmp{destination.suffix}")


def _save_image_atomic(image: Image.Image, destination: Path, *, jpeg_quality: int) -> None:
    temp = _temporary_image_path(destination)
    try:
        _save_image(image, temp, jpeg_quality=jpeg_quality)
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def _save_image(image: Image.Image, destination: Path, *, jpeg_quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        kwargs = {"quality": jpeg_quality, "subsampling": 0}
    image.save(destination, **kwargs)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def _save_debug(image: Image.Image, result: CleanImageResult, path: Path) -> None:
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    for detection in result.detections:
        x, y, w, h = detection.box
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=3)
        draw.text(
            (x + 4, y + 4),
            f"{detection.name} {detection.confidence:.2f}",
            fill=(255, 0, 0),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(path)
