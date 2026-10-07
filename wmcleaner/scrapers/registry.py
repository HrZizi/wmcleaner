from __future__ import annotations

from .base import ScrapeError, SourceScraper
from .baozimh import BaozimhScraper

# Add future site adapters here.  Source selection stays centralized so the CLI
# never needs site-specific conditionals.
SOURCES: tuple[type[SourceScraper], ...] = (
    BaozimhScraper,
)


def source_for(url: str) -> SourceScraper:
    for source_type in SOURCES:
        if source_type.supports(url):
            return source_type()
    supported = ", ".join(source.name for source in SOURCES)
    raise ScrapeError(f"Unsupported source URL. Available sources: {supported}")
