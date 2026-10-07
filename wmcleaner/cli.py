from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .cleaning import CleanSettings, Cleaner
from .detector import combine_masks, detect_watermarks
from .inpaint import LamaUnavailable, anime_lama_cached, lama_available
from .scrapers import ScrapeError, discover_series, download_source, select_chapters
from .pipeline import scrape_and_clean

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image type: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _destination(input_root: Path, src: Path, output: Path | None, suffix: str = "") -> Path:
    if input_root.is_file():
        if output is None:
            return src.with_name(src.stem + (suffix or ".cleaned") + src.suffix)
        if output.suffix:
            return output
        return output / src.name

    root = output or input_root.with_name(input_root.name + (suffix or "_cleaned"))
    rel = src.relative_to(input_root)
    return root / rel


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def _save_debug(image: Image.Image, detections, path: Path) -> None:
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    for d in detections:
        x, y, w, h = d.box
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=3)
        draw.text((x + 4, y + 4), f"{d.name} {d.confidence:.2f}", fill=(255, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(path)


def cmd_detect(args: argparse.Namespace) -> int:
    src_root = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve() if args.output else None
    files = _images(src_root)
    found = 0

    for src in files:
        image = Image.open(src).convert("RGB")
        det = detect_watermarks(
            image,
            aggressive=args.aggressive,
            dilation=args.dilation,
            min_confidence=args.min_confidence,
        )
        mask = combine_masks(det, (image.height, image.width))
        if det:
            found += 1

        dst = _destination(src_root, src, out_root, "_masks")
        dst = dst.with_suffix(".png")
        _save_mask(mask, dst)
        if args.debug:
            debug = dst.with_name(dst.stem + ".debug.png")
            _save_debug(image, det, debug)

        labels = ", ".join(f"{d.name}:{d.confidence:.2f}" for d in det) or "none"
        print(f"[detect] {src.name}: {labels} -> {dst}")

    print(f"Detected watermark(s) in {found}/{len(files)} image(s).")
    return 0


def _clean_settings_from_args(args: argparse.Namespace) -> CleanSettings:
    return CleanSettings(
        backend=args.backend,
        device=args.device,
        aggressive=args.aggressive,
        dilation=args.dilation,
        min_confidence=args.min_confidence,
        comic_overlap_backend=args.comic_overlap_backend,
        comic_overlap_threshold=args.comic_overlap_threshold,
        comic_dense_overlap_threshold=args.comic_dense_overlap_threshold,
        logo_mask_style=args.logo_mask_style,
        logo_mask_padding=args.logo_mask_padding,
        legacy_banner_mode=args.legacy_banner_mode,
    )


def cmd_clean(args: argparse.Namespace) -> int:
    src_root = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve() if args.output else None
    files = _images(src_root)

    try:
        cleaner = Cleaner(_clean_settings_from_args(args))
    except LamaUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"Backend: {cleaner.primary_backend} | AI installed: {'yes' if lama_available() else 'no'}")
    if cleaner.primary_backend == "lama":
        print("Note: the first LaMa run may download its model once.")

    processed = 0
    skipped = 0
    for src in files:
        image = Image.open(src).convert("RGB")
        cleaned = cleaner.clean(image)

        if not cleaned.cleaned:
            skipped += 1
            qa = tuple(getattr(cleaned, "qa_reasons", ()))
            if cleaned.detections and qa:
                print(f"[qa]    {src.name}: cleanup held: {', '.join(qa)}")
            if args.copy_unmatched:
                result = image
            else:
                if not (cleaned.detections and qa):
                    print(f"[skip]  {src.name}: no known watermark detected")
                continue
        else:
            if cleaned.used_comic_fallback:
                if getattr(cleaned, "used_legacy_banner_mask", False):
                    detail = getattr(cleaned, "legacy_banner_mode_used", None) or "unknown"
                    deblended = getattr(cleaned, "legacy_deblended_fraction", 0.0)
                    ai_fraction = getattr(cleaned, "legacy_ai_mask_fraction", 0.0)
                    structural = getattr(cleaned, "legacy_structural_pixel_count", 0)
                    boundary = getattr(cleaned, "legacy_boundary_fraction", 0.0)
                    fit_space = getattr(cleaned, "legacy_fit_space", None)
                    alpha = getattr(cleaned, "legacy_alpha", None)
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
                    print(f"[legacy] {src.name}: anime-lama {detail} {metrics}{fit}")
                else:
                    tag = "comic-dense" if cleaned.used_dense_logo_mask else "comic"
                    suffix = (
                        f" | dense-{cleaned.dense_repair_mode}-repair"
                        if cleaned.dense_repair_mode
                        else ""
                    )
                    print(
                        f"[{tag}] {src.name}: anime-lama "
                        f"structural-overlap={cleaned.overlap_score:.3f}{suffix}"
                    )
            result = cleaned.image
            processed += 1

        dst = _destination(src_root, src, out_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {}
        if dst.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs = {"quality": args.jpeg_quality, "subsampling": 0}
        result.save(dst, **save_kwargs)

        if args.save_masks:
            mask_path = dst.with_name(dst.stem + ".mask.png")
            _save_mask(cleaned.mask, mask_path)
        if args.debug:
            debug = dst.with_name(dst.stem + ".debug.png")
            _save_debug(image, cleaned.detections, debug)

        labels = ", ".join(f"{d.name}:{d.confidence:.2f}" for d in cleaned.detections) or "none"
        print(f"[clean] {src.name}: {labels} -> {dst}")

    print(f"Done. Cleaned {processed}; unmatched {skipped}; total {len(files)}.")
    return 0




def _prompt_required(value: str | None, label: str) -> str:
    if value and value.strip():
        return value.strip()
    while True:
        entered = input(f"{label}: ").strip()
        if entered:
            return entered
        print(f"{label} is required.")


def _print_chapter_listing(source_url: str, selector: str) -> int:
    try:
        series = discover_series(source_url)
        selected = select_chapters(series.chapters, selector)
    except ScrapeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    selected_indices = {chapter.index for chapter in selected}
    print(f'{series.title} | {len(series.chapters)} chapter(s)')
    for chapter in series.chapters:
        marker = "*" if chapter.index in selected_indices else " "
        print(
            f"{marker} [{chapter.index:04d}] {chapter.title} "
            f"(slug={chapter.slug}, id={chapter.id or '-'})"
        )
    print(f"Selected: {len(selected)}/{len(series.chapters)}")
    return 0


def cmd_scrape(args: argparse.Namespace) -> int:
    source_url = _prompt_required(args.source_url, "Source URL")
    if args.list_chapters:
        return _print_chapter_listing(source_url, args.chapters)

    output_value = _prompt_required(args.scrape_output, "Output directory")
    output_dir = Path(output_value).expanduser().resolve()

    try:
        result = download_source(
            source_url,
            output_dir,
            chapter_selector=args.chapters,
            delay=args.delay,
            overwrite=args.overwrite,
            write_manifest=not args.no_manifest,
        )
    except ScrapeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if result.kind == "series":
        print(
            f"Done. {result.chapter_count} chapter(s), {len(result.files)} page(s) "
            f"available under {output_dir}"
        )
    else:
        print(f"Done. {len(result.files)} page(s) available in {output_dir}")
    return 0


def cmd_scrape_clean(args: argparse.Namespace) -> int:
    source_url = _prompt_required(args.source_url, "Source URL")
    if args.list_chapters:
        return _print_chapter_listing(source_url, args.chapters)

    output_value = _prompt_required(args.scrape_output, "Output directory")
    output_dir = Path(output_value).expanduser().resolve()
    clean_output = (
        Path(args.clean_output).expanduser().resolve() if args.clean_output else None
    )

    settings = _clean_settings_from_args(args)
    try:
        result = scrape_and_clean(
            source_url,
            output_dir,
            chapter_selector=args.chapters,
            delay=args.delay,
            overwrite=args.overwrite,
            write_manifest=not args.no_manifest,
            clean_settings=settings,
            clean_output=clean_output,
            in_place=args.in_place,
            replaced_dir_name=args.replaced_dir,
            save_masks=args.save_masks,
            debug=args.debug,
            jpeg_quality=args.jpeg_quality,
        )
    except (ScrapeError, LamaUnavailable) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    layout = "in-place" if result.in_place else f"separate tree: {result.clean_root}"
    print(
        f"Done. Scraped {result.download.chapter_count} chapter(s), "
        f"processed {len(result.pages)} page(s); cleaned {result.cleaned_count}, "
        f"unchanged {result.unchanged_count}. Output: {layout}"
    )
    if result.in_place and result.backup_count:
        print(
            f"Backed up {result.backup_count} replaced original page(s) "
            f"inside per-chapter {args.replaced_dir!r} folders."
        )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    import platform
    print(f"Python: {platform.python_version()}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"LaMa package: {'installed' if lama_available() else 'not installed'}")
    print(f"Anime-LaMa model: {'cached' if anime_lama_cached() else 'downloads on first use'}")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"PyTorch: unavailable ({e})")
    try:
        blank = Image.new("RGB", (850, 1000), "white")
        _ = detect_watermarks(blank)
        print("Detector templates: OK")
    except Exception as e:
        print(f"Detector templates: ERROR ({e})")
        return 2
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wmclean",
        description="Download supported comic pages and detect/remove known watermarks with LaMa or OpenCV inpainting.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("input", help="Image file or directory (directories are processed recursively).")
        sp.add_argument("-o", "--output", help="Output image/file directory. Defaults beside the input.")
        sp.add_argument("--aggressive", action="store_true", help="Mask the full overlay box instead of only watermark foreground pixels.")
        sp.add_argument("--dilation", type=int, default=1, help="Expand the mask by N pixels. Default: 1")
        sp.add_argument("--min-confidence", type=float, default=None, help="Override the built-in detection threshold (0..1).")
        sp.add_argument("--debug", action="store_true", help="Write debug images with detected boxes.")

    def add_clean_options(sp):
        sp.add_argument("--backend", choices=["auto", "lama", "anime-lama", "opencv"], default="auto", help="Inpainting backend. Default: auto")
        sp.add_argument(
            "--comic-overlap-backend",
            choices=["auto", "off", "anime-lama"],
            default="auto",
            help=(
                "For detected manga text/borders crossing a modern Baozimh watermark, "
                "switch that page to the anime/manga-finetuned LaMa model. "
                "auto enables it when the primary backend is LaMa; default: auto"
            ),
        )
        sp.add_argument(
            "--comic-overlap-threshold",
            type=float,
            default=0.035,
            help="Structural-overlap score required for the manga-specific fallback. Default: 0.035",
        )
        sp.add_argument(
            "--comic-dense-overlap-threshold",
            type=float,
            default=0.18,
            help=(
                "For very high structural-overlap pages, stop punching dark pixels back out "
                "of the solid logo mask. This prevents character-shaped mask holes on dark "
                "artwork that can make LaMa redraw watermark text. Default: 0.18"
            ),
        )
        sp.add_argument(
            "--logo-mask-style",
            choices=["solid", "precise"],
            default="solid",
            help="Modern Baozimh cleanup mask: solid avoids text-shaped LaMa smudges; precise keeps the old foreground mask. Default: solid",
        )
        sp.add_argument(
            "--logo-mask-padding",
            type=int,
            default=3,
            help="Extra pixels around the solid modern-logo cleanup region. Default: 3",
        )
        sp.add_argument(
            "--legacy-banner-mode",
            choices=["auto", "deblend", "precise", "solid"],
            default="auto",
            help=(
                "Legacy translucent banner cleanup: coherent v0.3.1.4-style repair with "
                "boundary polish, experimental recovery-first deblend, precise "
                "foreground-only repair, or unpolished solid compatibility. Default: auto"
            ),
        )
        sp.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device for LaMa. Default: auto")
        sp.add_argument("--save-masks", action="store_true", help="Save the generated mask beside every cleaned image.")
        sp.add_argument("--jpeg-quality", type=int, default=97, help="JPEG output quality. Default: 97")

    d = sub.add_parser("detect", help="Detect watermarks and write masks; do not modify source images.")
    common(d)
    d.set_defaults(func=cmd_detect)

    c = sub.add_parser("clean", help="Detect and remove watermarks.")
    common(c)
    add_clean_options(c)
    c.add_argument("--copy-unmatched", action="store_true", help="Copy images for which no watermark is detected into the output tree.")
    c.set_defaults(func=cmd_clean)

    scrape = sub.add_parser(
        "scrape",
        help="Download a supported chapter URL or all/selected chapters from a series detail URL.",
    )
    scrape.add_argument("source_url", nargs="?", help="Chapter or series/detail URL. Prompted when omitted.")
    scrape.add_argument(
        "scrape_output",
        nargs="?",
        help="Output directory. For a series, chapter subfolders are created below it. Prompted when omitted.",
    )
    scrape.add_argument(
        "--chapters",
        default="all",
        metavar="SELECTOR",
        help=(
            "Series chapter subset by 1-based list index: all, 1-30, 1,3,7-10, or 20-. "
            "Use --list-chapters to preview indices. Default: all"
        ),
    )
    scrape.add_argument(
        "--list-chapters",
        action="store_true",
        help="List chapters from a series/detail URL, mark the current --chapters selection, and exit.",
    )
    scrape.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between image downloads and chapter transitions. Default: 1.0")
    scrape.add_argument("--overwrite", action="store_true", help="Overwrite already-downloaded numbered page files.")
    scrape.add_argument("--no-manifest", action="store_true", help="Do not write chapter or series manifest JSON files.")
    scrape.set_defaults(func=cmd_scrape)

    sc = sub.add_parser(
        "scrape-clean",
        help="Download a supported chapter/series and immediately clean the downloaded pages.",
    )
    sc.add_argument("source_url", nargs="?", help="Chapter or series/detail URL. Prompted when omitted.")
    sc.add_argument(
        "scrape_output",
        nargs="?",
        help="Scrape/original output directory. Prompted when omitted and created as needed.",
    )
    sc.add_argument(
        "--chapters",
        default="all",
        metavar="SELECTOR",
        help="Series subset by 1-based source-list index: all, 1-30, 1,3,7-10, -5, or 20-. Default: all",
    )
    sc.add_argument("--list-chapters", action="store_true", help="List series chapter indices/titles and exit without downloading.")
    sc.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between image downloads and chapter transitions. Default: 1.0")
    sc.add_argument("--overwrite", action="store_true", help="Overwrite already-downloaded numbered page files before cleaning.")
    sc.add_argument("--no-manifest", action="store_true", help="Do not write scrape, series, or scrape-clean manifest JSON files.")
    sc.add_argument(
        "--clean-output",
        help="Separate cleaned-tree directory. Defaults to <scrape-output>_cleaned. Cannot be combined with --in-place.",
    )
    sc.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Put cleaned pages back at their original scraped paths. Before each replacement, "
            "move that original page into a per-chapter backup subfolder."
        ),
    )
    sc.add_argument(
        "--replaced-dir",
        default="_replaced",
        metavar="NAME",
        help="Backup subfolder used by --in-place. Default: _replaced",
    )
    sc.add_argument("--aggressive", action="store_true", help="Mask the full overlay box instead of only watermark foreground pixels.")
    sc.add_argument("--dilation", type=int, default=1, help="Expand the mask by N pixels. Default: 1")
    sc.add_argument("--min-confidence", type=float, default=None, help="Override the built-in detection threshold (0..1).")
    sc.add_argument("--debug", action="store_true", help="Write debug images beside cleaned pages.")
    add_clean_options(sc)
    sc.set_defaults(func=cmd_scrape_clean)

    doc = sub.add_parser("doctor", help="Show installation, CUDA, and backend status.")
    doc.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
