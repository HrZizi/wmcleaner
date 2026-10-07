from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wmcleaner.scrapers.baozimh import BaozimhScraper
from wmcleaner.scrapers.base import ChapterRef, PageImage, SeriesInfo
from wmcleaner.scrapers.downloader import (
    _download_series_with_session,
    chapter_directory_name,
    select_chapters,
)


class _Response:
    def __init__(self, *, text: str = "", payload=None) -> None:
        self.text = text
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses: dict[str, _Response] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[str] = []

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        return self.responses[url]


class BaozimhSeriesTests(unittest.TestCase):
    def test_series_url_detection_distinguishes_detail_and_chapter(self) -> None:
        self.assertTrue(
            BaozimhScraper.is_series_url(
                "https://baozimh.org/manga/sudikanwodeyanshenzhujianbianzhi"
            )
        )
        self.assertFalse(
            BaozimhScraper.is_series_url(
                "https://baozimh.org/manga/sudikanwodeyanshenzhujianbianzhi/29694-031404030-0"
            )
        )

    def test_discover_org_series_uses_bookmark_mid_and_mode_all(self) -> None:
        source = "https://baozimh.org/manga/series-slug"
        list_url = "https://v2.apikk.top/api/v2/manga/get?mid=29694&mode=all"
        session = _Session(
            {
                source: _Response(
                    text='''
                        <div id="info"><h1>眼神逐渐变质</h1></div>
                        <div id="bookmarkData" data-mid="29694"></div>
                    '''
                ),
                list_url: _Response(
                    payload={
                        "data": {
                            "chapters": [
                                {
                                    "id": 101,
                                    "attributes": {"slug": "chapter-a", "title": "第01话"},
                                },
                                {
                                    "id": 102,
                                    "attributes": {"slug": "chapter-b", "title": "第02话"},
                                },
                            ]
                        }
                    }
                ),
            }
        )

        series = BaozimhScraper().discover_series(session, source)

        self.assertEqual(series.title, "眼神逐渐变质")
        self.assertEqual(series.slug, "series-slug")
        self.assertEqual(len(series.chapters), 2)
        self.assertEqual(series.chapters[0].index, 1)
        self.assertEqual(series.chapters[0].id, "101")
        self.assertEqual(series.chapters[0].url, "https://baozimh.org/manga/series-slug/chapter-a")
        self.assertEqual(series.chapters[1].title, "第02话")
        self.assertEqual(session.calls, [source, list_url])


class ChapterSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chapters = tuple(
            ChapterRef(i, str(i), f"slug-{i}", f"Chapter {i}", f"https://x/{i}")
            for i in range(1, 11)
        )

    def test_all_selector(self) -> None:
        self.assertEqual([c.index for c in select_chapters(self.chapters, "all")], list(range(1, 11)))

    def test_mixed_ranges_are_deduped_and_kept_in_source_order(self) -> None:
        selected = select_chapters(self.chapters, "8,2-4,3,10")
        self.assertEqual([c.index for c in selected], [2, 3, 4, 8, 10])

    def test_open_ranges(self) -> None:
        self.assertEqual([c.index for c in select_chapters(self.chapters, "-3")], [1, 2, 3])
        self.assertEqual([c.index for c in select_chapters(self.chapters, "8-")], [8, 9, 10])

    def test_out_of_range_selector_fails(self) -> None:
        with self.assertRaisesRegex(Exception, "only 10 chapters"):
            select_chapters(self.chapters, "11")

    def test_chapter_directory_is_windows_safe(self) -> None:
        chapter = ChapterRef(7, "x", "slug", 'A:B/C*D? "test"', "https://x")
        self.assertEqual(chapter_directory_name(chapter), '0007_A_B_C_D_ _test_')


class SeriesDownloadLayoutTests(unittest.TestCase):
    def test_selected_chapters_keep_source_indices_and_write_root_manifest(self) -> None:
        chapters = tuple(
            ChapterRef(i, str(i), f"slug-{i}", f"Chapter {i}", f"https://example/ch-{i}")
            for i in range(1, 5)
        )
        series = SeriesInfo("Series", "series", "https://example/series", chapters)

        class FakeScraper:
            name = "fake"

            def discover_series(self, session, source_url):
                return series

            def discover_pages(self, session, source_url):
                return [PageImage(f"https://images/{source_url.rsplit('-', 1)[-1]}.webp", "https://example")]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_files: list[Path] = []

            def fake_download(scraper, session, url, output_dir, **kwargs):
                output_dir.mkdir(parents=True, exist_ok=True)
                p = output_dir / "0001.webp"
                p.write_bytes(b"x")
                fake_files.append(p)
                return [p]

            with patch(
                "wmcleaner.scrapers.downloader._download_chapter_with_session",
                side_effect=fake_download,
            ):
                result = _download_series_with_session(
                    FakeScraper(),
                    object(),
                    "https://example/series",
                    root,
                    chapter_selector="2-3",
                    delay=0,
                    overwrite=False,
                    write_manifest=True,
                )

            self.assertEqual([c.index for c in result.selected], [2, 3])
            self.assertTrue((root / "0002_Chapter 2").is_dir())
            self.assertTrue((root / "0003_Chapter 3").is_dir())
            self.assertFalse((root / "0001_Chapter 1").exists())
            self.assertTrue((root / "series-manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
