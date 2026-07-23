from __future__ import annotations

import socket

import pytest

from src.config.settings import Settings
from src.ingestion.web_loader import (
    WebIngestionError,
    WebIngestionService,
    extract_public_page,
    normalize_public_url,
)

ARTICLE_BODY = (
    "市は交通空白地域で予約制バスの実証運行を開始する。"
    "対象は住民全員で、期間は六か月を予定している。"
    "利用実績を検証し、本格導入の可否を判断する。"
)

ARTICLE_HTML = """
<!doctype html>
<html lang="ja">
<head>
  <meta property="og:title" content="公共交通の実証運行を開始">
  <meta property="og:description" content="市が実証運行の計画を発表した。">
  <meta property="og:site_name" content="Example News">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "NewsArticle",
    "headline": "公共交通の実証運行を開始",
    "datePublished": "2026-07-24T10:00:00+09:00",
    "articleBody": "__ARTICLE_BODY__",
    "comment": [{
      "@type": "Comment",
      "identifier": "comment-1",
      "text": "電話予約にも対応してほしい。",
      "dateCreated": "2026-07-24T10:10:00+09:00",
      "upvoteCount": 12,
      "author": {"name": "保存してはいけない名前"}
    }]
  }
  </script>
</head>
<body><article><p>市は交通空白地域で予約制バスを運行する。</p></article></body>
</html>
""".replace("__ARTICLE_BODY__", ARTICLE_BODY)


def test_extract_public_page_uses_structured_article_and_comments() -> None:
    result = extract_public_page(ARTICLE_HTML, "https://example.com/news/1")
    assert result.article.title == "公共交通の実証運行を開始"
    assert "本格導入" in result.article.body
    assert result.comments[0].text == "電話予約にも対応してほしい。"
    assert result.comments[0].empathy_count == 12
    assert "author" not in result.comments[0].model_dump()


def test_normalize_public_url_rejects_local_addresses() -> None:
    with pytest.raises(WebIngestionError):
        normalize_public_url("http://localhost/private")
    with pytest.raises(WebIngestionError):
        normalize_public_url("http://127.0.0.1/private")


class FakeHttp:
    def get_text(self, url: str, *, check_html: bool = True) -> tuple[str, str]:
        del check_html
        if url.endswith("/robots.txt"):
            return "User-agent: *\nDisallow: /private", url
        return ARTICLE_HTML, url


def test_service_respects_robots_txt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    service = WebIngestionService(Settings(), http_client=FakeHttp())  # type: ignore[arg-type]
    with pytest.raises(WebIngestionError, match="robots.txt"):
        service.fetch("https://example.com/private/article")
