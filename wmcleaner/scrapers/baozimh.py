from __future__ import annotations

import json
import os
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from .base import ChapterRef, PageImage, ScrapeError, SeriesInfo, SourceScraper


class BaozimhScraper(SourceScraper):
    """Downloader for Baozimh chapter-reader pages.

    Baozimh currently has two unrelated reader families:

    * ``baozimh.org`` uses the Goda-style reader/API.
    * ``baozimh.com`` still uses the older Baozi reader markup/app mirror.

    Keep the source-specific logic in this adapter so future sites can be added
    as separate ``SourceScraper`` implementations in ``registry.py``.
    """

    name = "baozimh"
    HOSTS = {
        "baozimh.org",
        "www.baozimh.org",
        "baozimh.com",
        "www.baozimh.com",
    }

    APP_BASE = "https://app.baozimh.com/baozimhapp"

    # baozimh.org migrated away from api-get-v*.mgsearcher.com.  The reader
    # currently exposes the active API host in its HTML; this is the fallback
    # used when that attribute is absent.
    ORG_API_BASE = "https://v2.apikk.top"
    # Current Goda/Baozimh image routes (reader v2.8.0+). The old
    # *.g-mh.online hosts were retired in August 2026. The site stores a
    # preferred route (c/t) in browser localStorage; a CLI has no browser
    # preference, so use the site's default c-route first and carry the
    # t-route as an automatic fallback for each page.
    ORG_IMAGE_HOSTS = {
        "c": {"line2": "https://c-nd2-1.6wm.top", "default": "https://c-nd3-1.6wm.top"},
        "t": {"line2": "https://t-nd2-1.6wm.top", "default": "https://t-nd3-1.6wm.top"},
    }
    ORG_IMAGE_ROUTE_DEFAULT = "c"

    # Chromium headless sends a "HeadlessChrome/..." UA.  baozimh.org omits
    # the reader island (``/assets/runtime/chapter-decoder.js`` defining
    # ``window.__cimg``) for that UA, so the decoder never appears.  Spoof a
    # normal desktop UA for headless Chromium instead.
    CHROMIUM_SPOOF_UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
    # Selenium may resolve "firefox" to the WindowsApps stub, which exits
    # immediately ("Process unexpectedly closed with status 0").  Prefer a
    # real install when one exists.
    FIREFOX_BINARIES = (
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
    )

    @classmethod
    def supports(cls, url: str) -> bool:
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        return parsed.scheme in {"http", "https"} and parsed.hostname in cls.HOSTS

    @classmethod
    def is_series_url(cls, url: str) -> bool:
        """Return True for the current baozimh.org ``/manga/<slug>`` detail page."""
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        if parsed.hostname not in {"baozimh.org", "www.baozimh.org"}:
            return False
        parts = [part for part in parsed.path.split("/") if part]
        return len(parts) == 2 and parts[0] == "manga"

    def discover_series(self, session: requests.Session, source_url: str) -> SeriesInfo:
        if not self.is_series_url(source_url):
            raise ScrapeError(
                "Expected a baozimh.org series URL shaped like /manga/<series>."
            )

        parsed = urlparse(source_url)
        site_origin = f"{parsed.scheme}://{parsed.netloc}"
        series_slug = [part for part in parsed.path.split("/") if part][1]

        response = self._get(session, source_url, referer=site_origin)
        soup = BeautifulSoup(response.text, "html.parser")
        bookmark = soup.select_one("#bookmarkData")
        drawer = soup.select_one("#chapterDrawerConfig")
        mid = self._data_attr(bookmark, "mid") or self._data_attr(drawer, "mid")
        api_base = (self._data_attr(drawer, "api-host") or self.ORG_API_BASE).rstrip("/")

        if not mid:
            raise ScrapeError(
                "Could not resolve the Baozimh manga id from the series page (#bookmarkData data-mid)."
            )

        list_url = f"{api_base}/api/v2/manga/get?mid={mid}&mode=all"
        payload = self._get_json(
            session,
            list_url,
            headers={"Origin": site_origin, "Referer": source_url},
        )
        try:
            raw_chapters = payload["data"]["chapters"]
        except (KeyError, TypeError):
            raise ScrapeError("Unexpected Baozimh v2 manga API response: missing chapter list.")
        if not isinstance(raw_chapters, list):
            raise ScrapeError("Unexpected Baozimh v2 manga API response: chapters is not a list.")

        chapters: list[ChapterRef] = []
        for raw in raw_chapters:
            if not isinstance(raw, dict):
                continue
            attrs = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
            slug = str(attrs.get("slug") or raw.get("slug") or "").strip().strip("/")
            if not slug:
                continue
            chapter_id = str(raw.get("id") or "").strip()
            title = str(attrs.get("title") or raw.get("title") or slug).strip() or slug
            chapters.append(
                ChapterRef(
                    index=len(chapters) + 1,
                    id=chapter_id,
                    slug=slug,
                    title=title,
                    url=urljoin(site_origin + "/", f"manga/{series_slug}/{slug}"),
                )
            )

        if not chapters:
            raise ScrapeError("Baozimh series API returned no usable chapters.")

        title_node = soup.select_one("#info h1") or soup.select_one("h1")
        series_title = title_node.get_text(" ", strip=True) if isinstance(title_node, Tag) else ""
        series_title = series_title or series_slug
        canonical_url = urljoin(site_origin + "/", f"manga/{series_slug}")
        return SeriesInfo(
            title=series_title,
            slug=series_slug,
            url=canonical_url,
            chapters=tuple(chapters),
        )

    def discover_pages(self, session: requests.Session, source_url: str) -> list[PageImage]:
        if not self.supports(source_url):
            raise ScrapeError(f"URL is not a supported Baozimh URL: {source_url}")
        if self.is_series_url(source_url):
            raise ScrapeError(
                "This is a Baozimh series/detail URL. Use series scraping instead of chapter-page discovery."
            )

        host = urlparse(source_url).hostname or ""
        api_error: ScrapeError | None = None

        if host in {"baozimh.org", "www.baozimh.org"}:
            try:
                pages = self._discover_org_api(session, source_url)
                if pages:
                    return self.dedupe(pages)
            except ScrapeError as exc:
                api_error = exc

            # Do not use app.baozimh.com as a fallback for baozimh.org.  They
            # are different reader families and the app host is no longer a
            # valid dependency for the .org site.
            try:
                pages = self._discover_markup(session, source_url, allow_app=False)
            except ScrapeError as exc:
                raise ScrapeError(
                    f"Baozimh current API path failed ({api_error}); markup fallback failed ({exc})"
                ) from exc
        else:
            pages = self._discover_markup(session, source_url, allow_app=True)

        if not pages:
            detail = f" API error: {api_error}" if api_error else ""
            raise ScrapeError(
                "No chapter images were found. Baozimh may have changed its reader markup/API." + detail
            )
        return self.dedupe(pages)

    def _discover_org_api(self, session: requests.Session, source_url: str) -> list[PageImage]:
        parsed = urlparse(source_url)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 3 or parts[0] != "manga":
            raise ScrapeError("Expected a baozimh.org chapter URL shaped like /manga/<series>/<chapter>.")

        site_origin = f"{parsed.scheme}://{parsed.netloc}"
        chapter_slug = parts[2]

        # The current reader puts the exact manga/chapter API ids on the
        # chapter page.  Prefer those ids; it avoids a second chapter-list
        # lookup and, importantly, avoids the retired mgsearcher endpoint.
        chapter_response = self._get(session, source_url, referer=site_origin)
        soup = BeautifulSoup(chapter_response.text, "html.parser")

        content = soup.select_one("#chapterContent")
        drawer = soup.select_one("#chapterDrawerConfig")
        bookmark = soup.select_one("#bookmarkData")

        ms = self._data_attr(content, "ms") or self._data_attr(drawer, "mid") or self._data_attr(bookmark, "mid")
        cs = self._data_attr(content, "cs") or self._data_attr(drawer, "current-chapter-id")
        api_base = self._data_attr(drawer, "api-host") or self.ORG_API_BASE
        api_base = api_base.rstrip("/")

        # Some variants expose only the manga id on first paint.  Resolve the
        # chapter id from the *current* v2 API in that case.
        if ms and not cs:
            list_url = f"{api_base}/api/v2/manga/get?mid={ms}&mode=all"
            chapter_json = self._get_json(
                session,
                list_url,
                headers={"Origin": site_origin, "Referer": source_url},
            )
            cs = self._chapter_id_for_slug(chapter_json, chapter_slug)

        if not ms or not cs:
            raise ScrapeError(
                "Could not resolve current Baozimh chapter parameters (data-ms/data-cs)."
            )

        info_url = f"{api_base}/api/v2/chapter/getinfo?m={ms}&c={cs}"
        info_json = self._get_json(
            session,
            info_url,
            headers={"Origin": site_origin, "Referer": source_url},
        )

        try:
            image_info = info_json["data"]["info"]["images"]
        except (KeyError, TypeError):
            raise ScrapeError("Unexpected Baozimh v2 chapter-info API response.")

        if not isinstance(image_info, dict):
            raise ScrapeError("Unexpected Baozimh v2 image metadata.")

        line = image_info.get("line")
        images = image_info.get("images")
        if isinstance(images, str):
            images = self._decode_org_images_with_browser(source_url, images)
        elif isinstance(images, dict) and isinstance(images.get("images"), list):
            images = images["images"]

        if not isinstance(images, list):
            raise ScrapeError("Baozimh v2 chapter-info API did not return a usable image list.")

        primary_base, alternate_base = self._org_image_bases(line)

        result: list[PageImage] = []
        for item in images:
            raw = self._image_path(item)
            if not raw:
                continue
            if raw.startswith(("http://", "https://")):
                # Respect a fully-qualified URL returned by the API. It may be a
                # newer CDN that this version does not know about yet.
                image_url = raw
                alternates: tuple[str, ...] = ()
            else:
                path = raw.lstrip("/")
                image_url = urljoin(primary_base + "/", path)
                alternates = (urljoin(alternate_base + "/", path),)
            result.append(PageImage(url=image_url, referer=site_origin, alternates=alternates))

        if not result:
            raise ScrapeError("Baozimh v2 chapter-info API returned no images.")
        return result


    @classmethod
    def _org_image_bases(cls, line: object) -> tuple[str, str]:
        line_key = "line2" if cls._as_int(line) == 2 else "default"
        primary_route = cls.ORG_IMAGE_ROUTE_DEFAULT
        alternate_route = "t" if primary_route == "c" else "c"
        primary = cls.ORG_IMAGE_HOSTS[primary_route][line_key]
        alternate = cls.ORG_IMAGE_HOSTS[alternate_route][line_key]
        return primary, alternate

    @staticmethod
    def _data_attr(node: Tag | None, name: str) -> str | None:
        if not isinstance(node, Tag):
            return None
        value = node.get(f"data-{name}")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _chapter_id_for_slug(payload: dict, chapter_slug: str) -> str | None:
        try:
            chapters = payload["data"]["chapters"]
        except (KeyError, TypeError):
            return None
        if not isinstance(chapters, list):
            return None
        for item in chapters:
            if not isinstance(item, dict):
                continue
            attrs = item.get("attributes")
            slug = attrs.get("slug") if isinstance(attrs, dict) else item.get("slug")
            if str(slug or "").strip("/") == chapter_slug:
                chapter_id = str(item.get("id", "")).strip()
                return chapter_id or None
        return None

    @staticmethod
    def _image_path(item: object) -> str | None:
        if isinstance(item, str):
            raw = item
        elif isinstance(item, dict):
            raw = next(
                (
                    item.get(key)
                    for key in ("url", "src", "img", "image", "path")
                    if isinstance(item.get(key), str) and str(item.get(key)).strip()
                ),
                None,
            )
        else:
            return None
        if not isinstance(raw, str):
            return None
        raw = raw.strip()
        return raw or None

    @staticmethod
    def _as_int(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _decode_org_images_with_browser(source_url: str, encoded: str) -> list:
        """Decode Baozimh's optional encoded image payload in the real reader JS.

        The site ships ``/assets/runtime/chapter-decoder.js`` and can change its
        algorithm independently of this project.  Rather than baking a fragile
        reverse-engineered decoder into wmcleaner, use Selenium only when the
        API returns an encoded string and execute the site's own ``__cimg.r``.
        """

        try:
            from selenium import webdriver
            from selenium.common.exceptions import WebDriverException
            from selenium.webdriver.support.ui import WebDriverWait
        except ImportError as exc:
            raise ScrapeError(
                "Baozimh returned an encoded image list, but Selenium is not installed. "
                "Run `pip install -e .` (or rerun setup_core.cmd) and retry."
            ) from exc

        errors: list[str] = []

        def drivers():
            firefox_options = webdriver.FirefoxOptions()
            firefox_options.add_argument("-headless")
            for candidate in BaozimhScraper.FIREFOX_BINARIES:
                if os.path.exists(candidate):
                    firefox_options.binary_location = candidate
                    break
            yield "Firefox", lambda: webdriver.Firefox(options=firefox_options)

            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument(f"--user-agent={BaozimhScraper.CHROMIUM_SPOOF_UA}")
            yield "Chrome", lambda: webdriver.Chrome(options=chrome_options)

            edge_options = webdriver.EdgeOptions()
            edge_options.add_argument("--headless=new")
            edge_options.add_argument("--disable-gpu")
            edge_options.add_argument(f"--user-agent={BaozimhScraper.CHROMIUM_SPOOF_UA}")
            yield "Edge", lambda: webdriver.Edge(options=edge_options)

        for browser_name, make_driver in drivers():
            driver = None
            try:
                driver = make_driver()
                driver.set_page_load_timeout(45)
                driver.set_script_timeout(30)
                driver.get(source_url)
                WebDriverWait(driver, 20).until(
                    lambda d: bool(
                        d.execute_script(
                            "return !!(window.__cimg && typeof window.__cimg.r === 'function');"
                        )
                    )
                )
                raw = driver.execute_async_script(
                    """
                    const payload = arguments[0];
                    const done = arguments[arguments.length - 1];
                    Promise.resolve(window.__cimg.r(payload))
                        .then(value => done(JSON.stringify({ok: true, value})))
                        .catch(error => done(JSON.stringify({ok: false, error: String(error)})));
                    """,
                    encoded,
                )
                decoded = json.loads(raw)
                if not decoded.get("ok"):
                    raise RuntimeError(decoded.get("error") or "decoder rejected payload")
                value = decoded.get("value")
                if not isinstance(value, list):
                    raise RuntimeError("decoder returned a non-list value")
                return value
            except Exception as exc:  # browser startup/driver/page/decoder failures
                errors.append(f"{browser_name}: {exc}")
            finally:
                if driver is not None:
                    try:
                        driver.quit()
                    except Exception:
                        pass

        detail = "; ".join(errors) or "no compatible browser could be started"
        raise ScrapeError(
            "Baozimh returned an encoded image list and browser decoding failed. "
            f"Tried Firefox, Chrome, and Edge. {detail}"
        )

    def _discover_markup(
        self,
        session: requests.Session,
        source_url: str,
        *,
        allow_app: bool,
    ) -> list[PageImage]:
        # Baozimh sometimes splits a chapter into reader sub-pages. Follow only
        # links explicitly labelled as "next page", never "next chapter".
        current_url: str | None = source_url
        visited: set[str] = set()
        pages: list[PageImage] = []

        while current_url:
            if current_url in visited:
                raise ScrapeError(f"Reader pagination loop detected at {current_url}")
            visited.add(current_url)

            response = self._get(session, current_url, referer=source_url)
            soup = BeautifulSoup(response.text, "html.parser")
            direct_pages = self._extract_reader_images(soup, current_url)

            if not direct_pages and allow_app:
                app_url = self._app_reader_url(current_url)
                app_response = self._get(session, app_url, referer=current_url)
                app_soup = BeautifulSoup(app_response.text, "html.parser")
                direct_pages = self._extract_app_images(app_soup, app_url)

            pages.extend(direct_pages)
            current_url = self._next_reader_page(soup, current_url)
        return pages

    @staticmethod
    def _get(session: requests.Session, url: str, referer: str) -> requests.Response:
        try:
            response = session.get(url, headers={"Referer": referer}, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise ScrapeError(f"Failed to fetch {url}: {exc}") from exc

    @staticmethod
    def _get_json(session: requests.Session, url: str, headers: dict[str, str]) -> dict:
        try:
            response = session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ScrapeError(f"Failed to fetch JSON {url}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ScrapeError(f"Expected a JSON object from {url}")
        return payload

    @classmethod
    def _app_reader_url(cls, url: str) -> str:
        parsed = urlparse(url)
        query = f"?{parsed.query}" if parsed.query else ""
        return f"{cls.APP_BASE}{parsed.path}{query}"

    @staticmethod
    def _extract_reader_images(soup: BeautifulSoup, page_url: str) -> list[PageImage]:
        selectors = [
            "amp-img[id^='chapter-img'][data-src]",
            ".chapter-img amp-img[data-src]",
            "amp-img[data-src]",
            ".chapter-img img[data-src]",
            ".chapter-img img[data-index][src]",
        ]
        nodes: list[Tag] = []
        for selector in selectors:
            found = [n for n in soup.select(selector) if isinstance(n, Tag)]
            if found:
                nodes = found
                break
        return BaozimhScraper._nodes_to_pages(nodes, page_url)

    @staticmethod
    def _extract_app_images(soup: BeautifulSoup, page_url: str) -> list[PageImage]:
        nodes = [n for n in soup.select(".chapter-img img[data-index]") if isinstance(n, Tag)]

        def key(node: Tag) -> tuple[int, str]:
            raw = str(node.get("data-index", ""))
            try:
                return int(raw), raw
            except ValueError:
                return 10**9, raw

        nodes.sort(key=key)
        return BaozimhScraper._nodes_to_pages(nodes, page_url)

    @staticmethod
    def _nodes_to_pages(nodes: list[Tag], referer: str) -> list[PageImage]:
        result: list[PageImage] = []
        for node in nodes:
            raw = node.get("data-src") or node.get("src")
            if not isinstance(raw, str):
                continue
            raw = raw.strip()
            if not raw or raw.startswith("data:"):
                continue
            result.append(PageImage(url=urljoin(referer, raw), referer=referer))
        return result

    @staticmethod
    def _next_reader_page(soup: BeautifulSoup, current_url: str) -> str | None:
        next_link = soup.select_one("#next-chapter")
        if not isinstance(next_link, Tag):
            return None
        label = next_link.get_text(" ", strip=True).lower()
        is_next_page = any(marker in label for marker in ("下一頁", "下一页", "next page"))
        href = next_link.get("href")
        if not is_next_page or not isinstance(href, str) or not href.strip():
            return None
        return urljoin(current_url, href.strip())
