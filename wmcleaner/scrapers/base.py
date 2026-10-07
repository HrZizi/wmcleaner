from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import requests


class ScrapeError(RuntimeError):
    """Raised when a source cannot be parsed or downloaded safely."""


@dataclass(frozen=True)
class PageImage:
    url: str
    referer: str
    alternates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChapterRef:
    """One chapter exposed by a series/detail page.

    ``index`` is the stable 1-based position returned by the source's chapter
    API/listing.  CLI chapter selectors operate on this index so users can run
    ``--list-chapters`` once and then request ranges such as ``1-30``.
    """

    index: int
    id: str
    slug: str
    title: str
    url: str


@dataclass(frozen=True)
class SeriesInfo:
    title: str
    slug: str
    url: str
    chapters: tuple[ChapterRef, ...]


class SourceScraper(ABC):
    """Interface for a website-specific scraper.

    All sources must support chapter-page scraping.  Series/detail-page
    discovery is optional and exposed through ``is_series_url`` /
    ``discover_series`` so future source adapters can add it without CLI
    site-specific branching.
    """

    name: str = "unknown"

    @classmethod
    @abstractmethod
    def supports(cls, url: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def discover_pages(self, session: requests.Session, source_url: str) -> list[PageImage]:
        raise NotImplementedError

    @classmethod
    def is_series_url(cls, url: str) -> bool:
        return False

    def discover_series(self, session: requests.Session, source_url: str) -> SeriesInfo:
        raise ScrapeError(f"{self.name} does not support series/detail-page scraping yet.")

    @staticmethod
    def dedupe(items: Iterable[PageImage]) -> list[PageImage]:
        seen: set[str] = set()
        result: list[PageImage] = []
        for item in items:
            if item.url in seen:
                continue
            seen.add(item.url)
            result.append(item)
        return result
