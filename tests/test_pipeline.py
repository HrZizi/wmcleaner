from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from wmcleaner.cleaning import CleanSettings
from wmcleaner.scrapers.downloader import SourceDownloadResult
from wmcleaner.pipeline import default_clean_root, scrape_and_clean


class _FakeCleaner:
    primary_backend = "lama"

    def __init__(self, cleaned_names: set[str]) -> None:
        self.cleaned_names = cleaned_names

    def clean(self, image: Image.Image):
        # The test writes the page name into pixel 0,0: red=1 means clean this page.
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        should_clean = int(arr[0, 0, 0]) == 1
        if not should_clean:
            return SimpleNamespace(
                image=image.copy(),
                mask=np.zeros((image.height, image.width), dtype=np.uint8),
                detections=[],
                cleaned=False,
                backend=None,
                overlap_score=0.0,
                used_comic_fallback=False,
            )

        out = Image.new("RGB", image.size, (10, 220, 10))
        detection = SimpleNamespace(name="baozimh_logo", confidence=0.95)
        mask = np.full((image.height, image.width), 255, dtype=np.uint8)
        return SimpleNamespace(
            image=out,
            mask=mask,
            detections=[detection],
            cleaned=True,
            backend="lama",
            overlap_score=0.01,
            used_comic_fallback=False,
        )


def _write_page(path: Path, *, marked: bool) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (12, 12), (1 if marked else 2, 20, 30))
    image.save(path)
    return path.read_bytes()


class ScrapeCleanPipelineTests(unittest.TestCase):
    def test_held_page_keeps_qa_and_source_bytes_in_both_layouts(self) -> None:
        for in_place in (False, True):
            with self.subTest(in_place=in_place), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "chapter"
                page = root / "0001.webp"
                original = _write_page(page, marked=False)
                download = SourceDownloadResult(kind="chapter", files=(page,), chapter_count=1, series=None)
                held = SimpleNamespace(cleaned=False,
                    detections=[SimpleNamespace(name="baozimh_logo", confidence=.99)],
                    overlap_score=.01, qa_reasons=("uncertain_registration",),
                    provenance={"plans": [{"route": "qa-hold"}]})
                from unittest.mock import Mock
                cleaner = Mock(primary_backend="lama")
                cleaner.clean.return_value = held
                with patch("wmcleaner.pipeline.download_source", return_value=download), patch(
                    "wmcleaner.pipeline.Cleaner", return_value=cleaner
                ):
                    result = scrape_and_clean("https://example/chapter", root, delay=0,
                        in_place=in_place, clean_settings=CleanSettings(backend="opencv"))
                record = result.pages[0]
                self.assertEqual(record.status, "unchanged-qa" if in_place else "copied-qa")
                self.assertEqual(record.qa_reasons, ("uncertain_registration",))
                self.assertTrue(record.detections)
                self.assertEqual(record.cleaner_provenance, held.provenance)
                self.assertEqual((result.clean_root / "0001.webp").read_bytes(), original)
                self.assertEqual(result.cleaned_count, 0)

    def test_separate_mode_keeps_originals_and_builds_complete_clean_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "series"
            chapter = root / "0001_Chapter 1"
            p1 = chapter / "0001.webp"
            p2 = chapter / "0002.webp"
            original1 = _write_page(p1, marked=True)
            original2 = _write_page(p2, marked=False)
            (chapter / "scrape-manifest.json").write_text('{"page_count": 2}', encoding="utf-8")
            (root / "series-manifest.json").write_text('{"chapters": []}', encoding="utf-8")
            download = SourceDownloadResult(
                kind="series", files=(p1, p2), chapter_count=1, series=None
            )

            with patch("wmcleaner.pipeline.download_source", return_value=download), patch(
                "wmcleaner.pipeline.Cleaner", return_value=_FakeCleaner({"0001.webp"})
            ):
                result = scrape_and_clean(
                    "https://example/series",
                    root,
                    delay=0,
                    clean_settings=CleanSettings(backend="opencv"),
                )

            clean_root = default_clean_root(root.resolve())
            self.assertEqual(result.clean_root, clean_root)
            self.assertEqual(p1.read_bytes(), original1)
            self.assertEqual(p2.read_bytes(), original2)
            self.assertTrue((clean_root / "0001_Chapter 1" / "0001.webp").is_file())
            self.assertEqual(
                (clean_root / "0001_Chapter 1" / "0002.webp").read_bytes(),
                original2,
            )
            self.assertTrue((clean_root / "0001_Chapter 1" / "scrape-manifest.json").is_file())
            self.assertTrue((clean_root / "series-manifest.json").is_file())
            manifest = json.loads((clean_root / "scrape-clean-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["layout"], "separate")
            self.assertEqual(manifest["cleaned_count"], 1)
            self.assertEqual(manifest["unchanged_count"], 1)
            self.assertEqual(len(manifest["pages"][0]["source_sha256"]), 64)
            self.assertEqual(len(manifest["pages"][0]["output_sha256"]), 64)
            self.assertEqual(manifest["pages"][0]["qa_reasons"], [])

    def test_in_place_mode_replaces_only_cleaned_pages_and_backs_up_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "chapter"
            p1 = root / "0001.webp"
            p2 = root / "0002.webp"
            original1 = _write_page(p1, marked=True)
            original2 = _write_page(p2, marked=False)
            download = SourceDownloadResult(
                kind="chapter", files=(p1, p2), chapter_count=1, series=None
            )

            with patch("wmcleaner.pipeline.download_source", return_value=download), patch(
                "wmcleaner.pipeline.Cleaner", return_value=_FakeCleaner({"0001.webp"})
            ):
                result = scrape_and_clean(
                    "https://example/chapter",
                    root,
                    delay=0,
                    in_place=True,
                    clean_settings=CleanSettings(backend="opencv"),
                )

            backup = root / "_replaced" / "0001.webp"
            self.assertTrue(backup.is_file())
            self.assertEqual(backup.read_bytes(), original1)
            self.assertNotEqual(p1.read_bytes(), original1)
            self.assertEqual(p2.read_bytes(), original2)
            self.assertFalse((root / "_replaced" / "0002.webp").exists())
            self.assertEqual(result.backup_count, 1)
            manifest = json.loads((root / "scrape-clean-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["layout"], "in-place")
            self.assertEqual(manifest["pages"][0]["backup"], "_replaced/0001.webp")

    def test_in_place_backup_collision_is_versioned_instead_of_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "chapter"
            p1 = root / "0001.webp"
            original = _write_page(p1, marked=True)
            backup_dir = root / "_replaced"
            backup_dir.mkdir(parents=True)
            existing = backup_dir / "0001.webp"
            existing.write_bytes(b"older-original")
            download = SourceDownloadResult(
                kind="chapter", files=(p1,), chapter_count=1, series=None
            )

            with patch("wmcleaner.pipeline.download_source", return_value=download), patch(
                "wmcleaner.pipeline.Cleaner", return_value=_FakeCleaner({"0001.webp"})
            ):
                scrape_and_clean(
                    "https://example/chapter",
                    root,
                    delay=0,
                    in_place=True,
                    clean_settings=CleanSettings(backend="opencv"),
                )

            self.assertEqual(existing.read_bytes(), b"older-original")
            self.assertEqual((backup_dir / "0001.2.webp").read_bytes(), original)

    def test_clean_output_and_in_place_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "chapter"
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                scrape_and_clean(
                    "https://example/chapter",
                    root,
                    clean_output=root / "clean",
                    in_place=True,
                )


if __name__ == "__main__":
    unittest.main()
