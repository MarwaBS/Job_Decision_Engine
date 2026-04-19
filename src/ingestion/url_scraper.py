"""URL scraper — best-effort only.

Architecture §3: this component is REAL but explicitly best-effort. Most
large job boards (LinkedIn, Indeed, Glassdoor) actively block scrapers.
The primary ingestion path is text paste; URL scraping is a convenience
for the minority of JDs on public pages.

Failure modes are surfaced via `UrlScrapeError` so the UI can gracefully
fall back to the paste path.
"""

from __future__ import annotations


class UrlScrapeError(RuntimeError):
    """Raised when a URL can't be fetched or its body can't be extracted.

    Caller is expected to catch this and ask the user to paste the JD text
    instead. Never silently fall back to an empty string — the scorer would
    then produce a low-confidence REVIEW verdict without any audit trail of
    why.
    """


def fetch_url(url: str, *, timeout_seconds: float = 10.0) -> str:
    """Fetch a URL and return its visible-text body.

    Args:
        url: Full HTTP/HTTPS URL.
        timeout_seconds: Hard timeout; the LinkedIn-block case fails fast.

    Returns:
        The page body as plain text, with HTML tags stripped.

    Raises:
        UrlScrapeError: Fetch failed, blocked, or body is empty.

    Implementation note: `requests` + `BeautifulSoup` are imported *inside*
    the function so the scorer and test suite don't need them installed.
    See `requirements.txt` for the production pins.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise UrlScrapeError(
            "url_scraper dependencies missing (requests, beautifulsoup4). "
            "Install them from requirements.txt."
        ) from e

    try:
        resp = requests.get(
            url,
            timeout=timeout_seconds,
            headers={
                # A polite user-agent. Most blocks will still reject us —
                # that's the "best-effort" claim the architecture makes.
                "User-Agent": (
                    "JobDecisionEngine/0.1 "
                    "(portfolio project; https://github.com/MarwaBS)"
                ),
            },
            allow_redirects=True,
        )
    except requests.RequestException as e:
        raise UrlScrapeError(f"fetch failed: {e}") from e

    if resp.status_code >= 400:
        raise UrlScrapeError(
            f"fetch returned HTTP {resp.status_code} — most likely blocked or expired."
        )

    soup = BeautifulSoup(resp.text, "html.parser")
    # Strip script/style/nav/footer — they are never part of the JD.
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    if not text.strip():
        raise UrlScrapeError("fetch succeeded but body is empty (likely JS-rendered page).")
    return text
