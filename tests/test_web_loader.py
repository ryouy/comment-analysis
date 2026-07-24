from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta

import pytest

from src.config.settings import Settings
from src.domain.models import Article, Comment
from src.ingestion.web_loader import (
    WebIngestionError,
    WebIngestionService,
    extract_public_page,
    normalize_public_url,
)
from src.ingestion.yahoo_selenium import (
    _comment_id_from_params,
    _comments_url,
    _integer,
    _parse_yahoo_datetime,
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


def test_extract_public_page_removes_related_links_from_dom_body() -> None:
    html = """
    <html><head><meta property="og:title" content="地域交通の計画"></head>
    <body><main><article>
      <div data-article-body>
        <p>市は交通空白地域で予約制バスの実証運行を開始すると発表した。</p>
        <p>計画の詳細は<a href="/source">市の資料</a>で公表され、期間は六か月を予定している。</p>
        <section class="RelatedArticles">
          <p><a href="/other">隣の市でも新しい交通サービスの運行が始まった</a></p>
        </section>
      </div>
    </article></main></body></html>
    """
    result = extract_public_page(html, "https://example.com/news/2")
    assert "予約制バス" in result.article.body
    assert "市の資料" in result.article.body
    assert "隣の市" not in result.article.body


def test_extract_public_page_removes_related_tail_from_structured_body() -> None:
    structured_body = (
        "市は予約制バスの実証運行を開始すると発表した。"
        "対象は交通空白地域の住民で、期間は六か月を予定している。"
        "\n【関連記事】 隣の市でも新交通サービス"
    )
    html = f"""
    <html><head>
      <meta property="og:title" content="地域交通の計画">
      <script type="application/ld+json">
      {{
        "@type": "NewsArticle",
        "articleBody": {json.dumps(structured_body, ensure_ascii=False)}
      }}
      </script>
    </head><body></body></html>
    """
    result = extract_public_page(html, "https://example.com/news/3")
    assert "六か月" in result.article.body
    assert "隣の市" not in result.article.body


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


def test_yahoo_comment_helpers_parse_current_values() -> None:
    reference = datetime.fromisoformat("2026-06-01T16:00:00+09:00")
    assert _integer("共感した\n1,234") == 1234
    assert (
        _comment_id_from_params("_cl_link:agbtn1;cmt_id:comment-123;")
        == "comment-123"
    )
    assert _parse_yahoo_datetime("6/2(火) 18:11", reference) == datetime.fromisoformat(
        "2026-06-02T18:11:00+09:00"
    )
    assert _parse_yahoo_datetime(
        "2026-06-02T18:11:00+09:00", reference
    ) == datetime.fromisoformat("2026-06-02T18:11:00+09:00")
    now = datetime.fromisoformat("2026-06-02T19:00:00+09:00")
    assert _parse_yahoo_datetime("15分前", reference, now) == now - timedelta(
        minutes=15
    )
    assert _comments_url("https://news.yahoo.co.jp/articles/abc", 3).endswith(
        "/articles/abc/comments?page=3"
    )


class AllowingFakeHttp:
    def get_text(self, url: str, *, check_html: bool = True) -> tuple[str, str]:
        del check_html
        if url.endswith("/robots.txt"):
            return "User-agent: *\nAllow: /", url
        return ARTICLE_HTML, url


class FakeYahooComments:
    def fetch(self, article: Article, limit: int) -> list[Comment]:
        assert limit == 50
        return [
            Comment(
                comment_id="y1",
                article_id=article.article_id,
                text="URLから取得したコメント",
                order_index=0,
            )
        ]


def test_service_merges_authorized_yahoo_comments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("183.79.250.251", 443))
        ],
    )
    service = WebIngestionService(
        Settings(),
        http_client=AllowingFakeHttp(),  # type: ignore[arg-type]
        yahoo_comment_provider=FakeYahooComments(),
    )
    result = service.fetch(
        "https://news.yahoo.co.jp/articles/example",
        comment_limit=50,
    )
    assert [comment.text for comment in result.comments] == [
        "URLから取得したコメント"
    ]
