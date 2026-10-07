from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import requests

from wmcleaner.scrapers.base import PageImage
from wmcleaner.scrapers.downloader import _download_one


class _ImageResponse:
    def __init__(self, body: bytes = b"webp-bytes", content_type: str = "image/webp") -> None:
        self.headers = {"Content-Type": content_type}
        self.body = body
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self.body

    def close(self) -> None:
        self.closed = True


class _FallbackSession:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.response = _ImageResponse()

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        if len(self.calls) == 1:
            raise requests.exceptions.SSLError("old/primary route TLS failure")
        return self.response


class DownloaderFallbackTests(unittest.TestCase):
    def test_download_retries_page_alternate_route(self) -> None:
        page = PageImage(
            url="https://c-nd2-1.6wm.top/hp/series/0/001.webp",
            referer="https://baozimh.org",
            alternates=("https://t-nd2-1.6wm.top/hp/series/0/001.webp",),
        )
        session = _FallbackSession()

        with tempfile.TemporaryDirectory() as tmp:
            path, used_url = _download_one(
                session, page, Path(tmp), 1, overwrite=False
            )
            self.assertEqual(path.name, "0001.webp")
            self.assertEqual(path.read_bytes(), b"webp-bytes")

        self.assertEqual(
            session.calls,
            [page.url, page.alternates[0]],
        )
        self.assertEqual(used_url, page.alternates[0])
        self.assertTrue(session.response.closed)


if __name__ == "__main__":
    unittest.main()
