from __future__ import annotations

import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup

from wmcleaner.scrapers.baozimh import BaozimhScraper
from wmcleaner.scrapers.registry import source_for


class _Response:
    def __init__(self, *, text: str = "", payload=None) -> None:
        self.text = text
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses: dict[str, _Response]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        return self.responses[url]


class BaozimhScraperTests(unittest.TestCase):
    def test_source_detection(self) -> None:
        self.assertEqual(source_for("https://baozimh.org/manga/foo/bar").name, "baozimh")
        self.assertEqual(source_for("https://www.baozimh.com/comic/chapter/foo/bar.html").name, "baozimh")

    def test_org_current_api_uses_ids_from_chapter_markup(self) -> None:
        source = "https://baozimh.org/manga/series-slug/chapter-slug"
        info_url = "https://v2.apikk.top/api/v2/chapter/getinfo?m=1234&c=77"
        session = _Session({
            source: _Response(text='''
                <div id="chapterContent" data-ms="1234" data-cs="77"></div>
                <div id="chapterDrawerConfig" data-api-host="https://v2.apikk.top"></div>
            '''),
            info_url: _Response(payload={"data": {"info": {"images": {
                "line": 1,
                "images": [
                    {"url": "/a/001.jpg"},
                    {"url": "https://cdn.example/002.webp"},
                ],
            }}}}),
        })

        pages = BaozimhScraper()._discover_org_api(session, source)
        self.assertEqual([p.url for p in pages], [
            "https://c-nd3-1.6wm.top/a/001.jpg",
            "https://cdn.example/002.webp",
        ])
        self.assertEqual(session.calls, [source, info_url])

    def test_org_line2_uses_line2_image_host(self) -> None:
        source = "https://baozimh.org/manga/series-slug/chapter-slug"
        info_url = "https://v2.apikk.top/api/v2/chapter/getinfo?m=29694&c=1444577"
        session = _Session({
            source: _Response(text='<div id="chapterContent" data-ms="29694" data-cs="1444577"></div>'),
            info_url: _Response(payload={"data": {"info": {"images": {
                "line": 2,
                "images": [{"url": "/chapter/001.jpg"}],
            }}}}),
        })

        pages = BaozimhScraper()._discover_org_api(session, source)
        self.assertEqual(pages[0].url, "https://c-nd2-1.6wm.top/chapter/001.jpg")
        self.assertEqual(pages[0].alternates, ("https://t-nd2-1.6wm.top/chapter/001.jpg",))

    def test_org_resolves_chapter_id_from_current_v2_api_when_markup_lacks_cs(self) -> None:
        source = "https://baozimh.org/manga/series-slug/chapter-slug"
        list_url = "https://v2.apikk.top/api/v2/manga/get?mid=1234&mode=all"
        info_url = "https://v2.apikk.top/api/v2/chapter/getinfo?m=1234&c=77"
        session = _Session({
            source: _Response(text='<div id="bookmarkData" data-mid="1234"></div>'),
            list_url: _Response(payload={"data": {"chapters": [
                {"id": 77, "attributes": {"slug": "chapter-slug"}},
            ]}}),
            info_url: _Response(payload={"data": {"info": {"images": {
                "line": 1,
                "images": [{"url": "/001.jpg"}],
            }}}}),
        })

        pages = BaozimhScraper()._discover_org_api(session, source)
        self.assertEqual(len(pages), 1)
        self.assertEqual(session.calls, [source, list_url, info_url])

    def test_org_encoded_images_use_browser_decoder_only_when_needed(self) -> None:
        source = "https://baozimh.org/manga/series-slug/chapter-slug"
        info_url = "https://v2.apikk.top/api/v2/chapter/getinfo?m=1234&c=77"
        session = _Session({
            source: _Response(text='<div id="chapterContent" data-ms="1234" data-cs="77"></div>'),
            info_url: _Response(payload={"data": {"info": {"images": {
                "line": 1,
                "images": "encrypted-payload",
            }}}}),
        })

        with patch.object(
            BaozimhScraper,
            "_decode_org_images_with_browser",
            return_value=[{"url": "/001.jpg"}, {"url": "/002.jpg"}],
        ) as decoder:
            pages = BaozimhScraper()._discover_org_api(session, source)

        decoder.assert_called_once_with(source, "encrypted-payload")
        self.assertEqual([p.url for p in pages], [
            "https://c-nd3-1.6wm.top/001.jpg",
            "https://c-nd3-1.6wm.top/002.jpg",
        ])

    def test_reader_image_extraction(self) -> None:
        html = """
        <html><body>
          <amp-img id="chapter-img-1" data-src="https://img.example/1.jpg"></amp-img>
          <amp-img id="chapter-img-2" data-src="/pages/2.webp"></amp-img>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        pages = BaozimhScraper._extract_reader_images(soup, "https://baozimh.org/manga/a/b")
        self.assertEqual([p.url for p in pages], [
            "https://img.example/1.jpg",
            "https://baozimh.org/pages/2.webp",
        ])

    def test_app_images_are_sorted_by_data_index(self) -> None:
        html = """
        <div class="chapter-img">
          <img data-index="2" data-src="https://img.example/2.jpg">
          <img data-index="1" data-src="https://img.example/1.jpg">
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        pages = BaozimhScraper._extract_app_images(soup, "https://app.baozimh.com/reader")
        self.assertEqual([p.url for p in pages], [
            "https://img.example/1.jpg",
            "https://img.example/2.jpg",
        ])

    def test_next_reader_page_only_follows_next_page_label(self) -> None:
        page = BeautifulSoup('<a id="next-chapter" href="/next">下一頁</a>', "html.parser")
        chapter = BeautifulSoup('<a id="next-chapter" href="/next">下一章</a>', "html.parser")
        self.assertEqual(
            BaozimhScraper._next_reader_page(page, "https://baozimh.org/manga/a/b"),
            "https://baozimh.org/next",
        )
        self.assertIsNone(BaozimhScraper._next_reader_page(chapter, "https://baozimh.org/manga/a/b"))


if __name__ == "__main__":
    unittest.main()
