from __future__ import annotations

import json
import mimetypes
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from .base import ChapterRef, PageImage, ScrapeError, SeriesInfo, SourceScraper
from .registry import source_for

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}
_CONTENT_TYPE_EXTS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
}
_WINDOWS_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class SeriesDownloadResult:
    series: SeriesInfo
    selected: tuple[ChapterRef, ...]
    chapter_directories: tuple[Path, ...]
    files: tuple[Path, ...]


@dataclass(frozen=True)
class SourceDownloadResult:
    kind: str
    files: tuple[Path, ...]
    chapter_count: int
    series: SeriesInfo | None = None


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        }
    )
    return session


def discover_series(source_url: str) -> SeriesInfo:
    scraper = source_for(source_url)
    if not scraper.is_series_url(source_url):
        raise ScrapeError("The supplied source URL is not a supported series/detail page.")
    return scraper.discover_series(make_session(), source_url)


def download_source(
    source_url: str,
    output_dir: Path,
    *,
    chapter_selector: str = "all",
    delay: float = 1.0,
    overwrite: bool = False,
    write_manifest: bool = True,
) -> SourceDownloadResult:
    """Download either one chapter URL or a complete/selected series URL."""
    scraper = source_for(source_url)
    session = make_session()

    if scraper.is_series_url(source_url):
        result = _download_series_with_session(
            scraper,
            session,
            source_url,
            output_dir,
            chapter_selector=chapter_selector,
            delay=delay,
            overwrite=overwrite,
            write_manifest=write_manifest,
        )
        return SourceDownloadResult(
            kind="series",
            files=result.files,
            chapter_count=len(result.selected),
            series=result.series,
        )

    if chapter_selector.strip().lower() not in {"", "all", "*"}:
        raise ScrapeError("--chapters can only be used with a series/detail URL.")
    files = _download_chapter_with_session(
        scraper,
        session,
        source_url,
        output_dir,
        delay=delay,
        overwrite=overwrite,
        write_manifest=write_manifest,
    )
    return SourceDownloadResult(kind="chapter", files=tuple(files), chapter_count=1)


def download_chapter(
    source_url: str,
    output_dir: Path,
    *,
    delay: float = 1.0,
    overwrite: bool = False,
    write_manifest: bool = True,
) -> list[Path]:
    if delay < 0:
        raise ValueError("delay must be >= 0")
    scraper = source_for(source_url)
    if scraper.is_series_url(source_url):
        raise ScrapeError("download_chapter received a series/detail URL; use download_source instead.")
    return _download_chapter_with_session(
        scraper,
        make_session(),
        source_url,
        output_dir,
        delay=delay,
        overwrite=overwrite,
        write_manifest=write_manifest,
    )


def _download_chapter_with_session(
    scraper: SourceScraper,
    session: requests.Session,
    source_url: str,
    output_dir: Path,
    *,
    delay: float,
    overwrite: bool,
    write_manifest: bool,
    manifest_extra: dict | None = None,
) -> list[Path]:
    if delay < 0:
        raise ValueError("delay must be >= 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    pages = scraper.discover_pages(session, source_url)
    print(f"Source: {scraper.name} | discovered {len(pages)} page(s)")

    downloaded: list[Path] = []
    manifest_pages: list[dict[str, str | int]] = []

    for index, page in enumerate(pages, start=1):
        if index > 1 and delay:
            time.sleep(delay)
        path, used_url = _download_one(session, page, output_dir, index, overwrite=overwrite)
        downloaded.append(path)
        manifest_pages.append({"index": index, "file": path.name, "url": used_url})
        print(f"[download] {index:04d}/{len(pages):04d} -> {path.name}")

    if write_manifest:
        manifest: dict = {
            "source": scraper.name,
            "source_url": source_url,
            "page_count": len(manifest_pages),
            "pages": manifest_pages,
        }
        if manifest_extra:
            manifest.update(manifest_extra)
        manifest_path = output_dir / "scrape-manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Manifest: {manifest_path}")

    return downloaded


def _download_series_with_session(
    scraper: SourceScraper,
    session: requests.Session,
    source_url: str,
    output_dir: Path,
    *,
    chapter_selector: str,
    delay: float,
    overwrite: bool,
    write_manifest: bool,
) -> SeriesDownloadResult:
    if delay < 0:
        raise ValueError("delay must be >= 0")

    series = scraper.discover_series(session, source_url)
    selected = select_chapters(series.chapters, chapter_selector)
    if not selected:
        raise ScrapeError("The chapter selector did not select any chapters.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f'Source: {scraper.name} | series "{series.title}" | '
        f"{len(series.chapters)} chapter(s) available; {len(selected)} selected"
    )

    all_files: list[Path] = []
    chapter_dirs: list[Path] = []
    manifest_chapters: list[dict] = []

    for selected_pos, chapter in enumerate(selected, start=1):
        if selected_pos > 1 and delay:
            # Keep a short pause between chapters too; Baozimh is known to rate-limit
            # aggressive sequential downloading.
            time.sleep(delay)

        chapter_dir = output_dir / chapter_directory_name(chapter)
        chapter_dirs.append(chapter_dir)
        print(
            f"[chapter] {selected_pos:04d}/{len(selected):04d} "
            f"[source #{chapter.index:04d}] {chapter.title} -> {chapter_dir.name}"
        )
        files = _download_chapter_with_session(
            scraper,
            session,
            chapter.url,
            chapter_dir,
            delay=delay,
            overwrite=overwrite,
            write_manifest=write_manifest,
            manifest_extra={
                "series_url": series.url,
                "series_title": series.title,
                "series_slug": series.slug,
                "chapter": {
                    "index": chapter.index,
                    "id": chapter.id,
                    "slug": chapter.slug,
                    "title": chapter.title,
                },
            },
        )
        all_files.extend(files)
        manifest_chapters.append(
            {
                "index": chapter.index,
                "id": chapter.id,
                "slug": chapter.slug,
                "title": chapter.title,
                "url": chapter.url,
                "directory": chapter_dir.name,
                "page_count": len(files),
            }
        )

    if write_manifest:
        series_manifest = {
            "source": scraper.name,
            "source_url": source_url,
            "series": {"title": series.title, "slug": series.slug, "url": series.url},
            "chapter_selector": chapter_selector or "all",
            "available_chapter_count": len(series.chapters),
            "downloaded_chapter_count": len(selected),
            "page_count": len(all_files),
            "chapters": manifest_chapters,
        }
        manifest_path = output_dir / "series-manifest.json"
        manifest_path.write_text(
            json.dumps(series_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Series manifest: {manifest_path}")

    return SeriesDownloadResult(
        series=series,
        selected=tuple(selected),
        chapter_directories=tuple(chapter_dirs),
        files=tuple(all_files),
    )


def select_chapters(chapters: tuple[ChapterRef, ...] | list[ChapterRef], selector: str) -> list[ChapterRef]:
    """Select chapters by 1-based site-list indices.

    Grammar examples: ``all``, ``1-30``, ``1,3,7-10``, ``-5`` (first five),
    and ``20-`` (chapter 20 through the end).  The resulting download order is
    always the source/site order, independent of token order or duplicates.
    """
    items = list(chapters)
    total = len(items)
    raw = (selector or "all").strip().lower()
    if raw in {"all", "*"}:
        return items
    if not raw:
        raise ScrapeError("Chapter selector cannot be empty. Use 'all' for every chapter.")

    selected_indices: set[int] = set()
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            raise ScrapeError(f"Invalid chapter selector {selector!r}: empty item.")
        if "-" in token:
            if token.count("-") != 1:
                raise ScrapeError(f"Invalid chapter range {token!r}.")
            start_text, end_text = token.split("-", 1)
            start = _selector_int(start_text, token) if start_text else 1
            end = _selector_int(end_text, token) if end_text else total
            if start > end:
                raise ScrapeError(f"Invalid chapter range {token!r}: start is after end.")
            _validate_selector_bound(start, total, token)
            _validate_selector_bound(end, total, token)
            selected_indices.update(range(start, end + 1))
        else:
            value = _selector_int(token, token)
            _validate_selector_bound(value, total, token)
            selected_indices.add(value)

    return [chapter for chapter in items if chapter.index in selected_indices]


def _selector_int(value: str, token: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ScrapeError(f"Invalid chapter selector item {token!r}; expected numbers/ranges.") from exc
    if parsed < 1:
        raise ScrapeError(f"Chapter indices are 1-based; invalid value {parsed}.")
    return parsed


def _validate_selector_bound(value: int, total: int, token: str) -> None:
    if value > total:
        raise ScrapeError(
            f"Chapter selector {token!r} references index {value}, but the series has only {total} chapters."
        )


def chapter_directory_name(chapter: ChapterRef) -> str:
    label = _safe_component(chapter.title) or _safe_component(chapter.slug) or "chapter"
    return f"{chapter.index:04d}_{label}"


def _safe_component(value: str, max_length: int = 90) -> str:
    value = _WINDOWS_INVALID.sub("_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    if not value:
        return ""
    # Avoid Windows reserved device names even when the project is run elsewhere.
    stem = value.split(".", 1)[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        value = "_" + value
    return value[:max_length].rstrip(" .")


def _download_one(
    session: requests.Session,
    page: PageImage,
    output_dir: Path,
    index: int,
    *,
    overwrite: bool,
) -> tuple[Path, str]:
    response: requests.Response | None = None
    used_url = page.url
    attempts: list[str] = []
    candidates = (page.url, *page.alternates)

    for candidate_index, candidate_url in enumerate(candidates):
        try:
            response = session.get(
                candidate_url,
                headers={
                    "Referer": page.referer,
                    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                },
                timeout=45,
                stream=True,
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type and not content_type.startswith("image/"):
                attempts.append(
                    f"{candidate_url}: returned {content_type!r} instead of an image"
                )
                response.close()
                response = None
                continue
            used_url = candidate_url
            if candidate_index:
                print(f"[fallback] page {index}: using alternate image route {urlparse(candidate_url).netloc}")
            break
        except requests.RequestException as exc:
            attempts.append(f"{candidate_url}: {exc}")
            if response is not None:
                response.close()
                response = None

    if response is None:
        detail = "; ".join(attempts)
        raise ScrapeError(f"Failed to download page {index}; all image routes failed ({detail})")

    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    ext = _extension(used_url, content_type)
    destination = output_dir / f"{index:04d}{ext}"
    if destination.exists() and not overwrite:
        print(f"[skip]     {destination.name} already exists")
        response.close()
        return destination, used_url

    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp.replace(destination)
    finally:
        response.close()
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return destination, used_url


def _extension(url: str, content_type: str) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in _IMAGE_EXTS:
        return ".jpg" if ext == ".jpeg" else ext
    if content_type in _CONTENT_TYPE_EXTS:
        return _CONTENT_TYPE_EXTS[content_type]
    guessed = mimetypes.guess_extension(content_type) if content_type else None
    return guessed or ".jpg"
