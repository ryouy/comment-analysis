from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag

from src.config.settings import Settings
from src.domain.models import Article, Comment
from src.ingestion.yahoo_selenium import (
    SeleniumYahooCommentProvider,
    YahooCommentFetchError,
    YahooCommentProvider,
)
from src.preprocessing.text import clean_text


class WebIngestionError(ValueError):
    """ユーザーへ安全に表示できるURL取得エラー。"""


@dataclass(frozen=True)
class WebIngestionResult:
    article: Article
    comments: list[Comment]
    warnings: list[str]


def normalize_public_url(url: str) -> str:
    value = url.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise WebIngestionError("httpまたはhttpsの公開URLを入力してください。")
    if parsed.username or parsed.password:
        raise WebIngestionError("認証情報を含むURLは使用できません。")
    try:
        port = parsed.port
    except ValueError as exc:
        raise WebIngestionError("URLのポート番号が不正です。") from exc
    if port not in {None, 80, 443}:
        raise WebIngestionError("標準ポート以外のURLは使用できません。")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".local"):
        raise WebIngestionError("ローカルネットワークのURLは使用できません。")
    try:
        addresses = {
            item[4][0] for item in socket.getaddrinfo(host, port or 443)
        }
    except socket.gaierror as exc:
        raise WebIngestionError("URLのホスト名を解決できません。") from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise WebIngestionError("公開インターネット以外のURLは使用できません。")
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path or "/", parsed.query, ""))


class SafeHttpClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": settings.web_fetch_user_agent,
                "Accept": "text/html,application/xhtml+xml",
            }
        )

    def get_text(self, url: str, *, check_html: bool = True) -> tuple[str, str]:
        current = normalize_public_url(url)
        for _ in range(6):
            try:
                response = self.session.get(
                    current,
                    timeout=self.settings.web_fetch_timeout_seconds,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as exc:
                raise WebIngestionError(f"URLを取得できません: {type(exc).__name__}") from exc
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("location")
                response.close()
                if not location:
                    raise WebIngestionError("リダイレクト先がありません。")
                current = normalize_public_url(urljoin(current, location))
                continue
            if response.status_code >= 400:
                status = response.status_code
                response.close()
                raise WebIngestionError(f"URL取得に失敗しました（HTTP {status}）。")
            content_type = response.headers.get("content-type", "").lower()
            if check_html and "text/html" not in content_type:
                response.close()
                raise WebIngestionError("HTMLページではありません。")
            chunks = []
            size = 0
            for chunk in response.iter_content(64 * 1024):
                size += len(chunk)
                if size > self.settings.web_fetch_max_bytes:
                    response.close()
                    raise WebIngestionError("ページサイズが取得上限を超えています。")
                chunks.append(chunk)
            encoding = response.encoding or response.apparent_encoding or "utf-8"
            response.close()
            return b"".join(chunks).decode(encoding, errors="replace"), current
        raise WebIngestionError("リダイレクト回数が上限を超えました。")


def _robots_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/robots.txt", "", ""))


def _json_documents(soup: BeautifulSoup) -> Iterable[Any]:
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            yield json.loads(script.get_text())
        except (json.JSONDecodeError, TypeError):
            continue


def _walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_json(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_json(nested)


def _first_text(soup: BeautifulSoup, selectors: list[str]) -> str | None:
    for selector in selectors:
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            raw_value = node.get("content", "") if node.name == "meta" else node.get_text(" ")
            if isinstance(raw_value, list):
                raw_value = " ".join(str(item) for item in raw_value)
            value = clean_text(str(raw_value or ""))
            if value:
                return value
    return None


def _article_body(soup: BeautifulSoup, documents: list[Any]) -> str:
    for document in documents:
        for item in _walk_json(document):
            kind = item.get("@type")
            if kind in {"Article", "NewsArticle", "ReportageNewsArticle"}:
                body = item.get("articleBody")
                if isinstance(body, str) and len(clean_text(body)) >= 50:
                    return clean_text(body)
    selectors = [
        "[itemprop='articleBody']",
        "article",
        ".article_body",
        "[class*='ArticleBody']",
        "main",
    ]
    candidates = []
    for selector in selectors:
        for node in soup.select(selector):
            if not isinstance(node, Tag):
                continue
            paragraphs = [clean_text(item.get_text(" ")) for item in node.select("p")]
            text = "\n".join(item for item in paragraphs if len(item) >= 15)
            if len(text) >= 50:
                candidates.append(text)
    if not candidates:
        raise WebIngestionError("記事本文を抽出できませんでした。")
    return max(candidates, key=len)


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _comment_values(soup: BeautifulSoup, documents: list[Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for document in documents:
        for item in _walk_json(document):
            if item.get("@type") != "Comment":
                continue
            text = item.get("text") or item.get("description")
            if isinstance(text, str):
                values.append(
                    {
                        "id": item.get("@id") or item.get("identifier"),
                        "text": text,
                        "posted_at": item.get("dateCreated") or item.get("datePublished"),
                        "empathy_count": item.get("upvoteCount"),
                    }
                )
    selectors = [
        "[itemprop='comment']",
        "[data-comment-id]",
        "article.comment",
        "li.comment",
        "[class*='CommentItem']",
    ]
    for selector in selectors:
        for node in soup.select(selector):
            if not isinstance(node, Tag):
                continue
            text_node = node.select_one(
                "[itemprop='text'], .comment-text, [class*='CommentText'], p"
            )
            text = clean_text(text_node.get_text(" ") if text_node else node.get_text(" "))
            if len(text) < 3:
                continue
            values.append(
                {
                    "id": node.get("data-comment-id") or node.get("id"),
                    "text": text,
                    "posted_at": node.get("datetime"),
                    "empathy_count": node.get("data-like-count"),
                }
            )
    return values


def extract_public_page(html: str, url: str) -> WebIngestionResult:
    soup = BeautifulSoup(html, "html.parser")
    documents = list(_json_documents(soup))
    for node in soup(["script", "style", "noscript", "template"]):
        if isinstance(node, Tag) and node.parent is not None:
            node.decompose()
    title = _first_text(
        soup,
        [
            "meta[property='og:title']",
            "meta[name='twitter:title']",
            "h1",
            "title",
        ],
    )
    if not title:
        raise WebIngestionError("記事タイトルを抽出できませんでした。")
    body = _article_body(soup, documents)
    summary = _first_text(
        soup, ["meta[property='og:description']", "meta[name='description']"]
    )
    source_name = _first_text(soup, ["meta[property='og:site_name']"])
    published_at = None
    for document in documents:
        for item in _walk_json(document):
            if item.get("@type") in {"Article", "NewsArticle", "ReportageNewsArticle"}:
                published_at = _parse_datetime(item.get("datePublished"))
                break
        if published_at:
            break
    article_id = hashlib.sha256(url.encode()).hexdigest()[:20]
    article = Article(
        article_id=article_id,
        source_url=url,
        source_name=source_name or urlsplit(url).hostname,
        title=title,
        summary=summary,
        body=body,
        published_at=published_at,
        fetched_at=datetime.now().astimezone(),
    )
    comments: list[Comment] = []
    seen = set()
    for value in _comment_values(soup, documents):
        text = clean_text(str(value["text"]))
        if not text or text in seen:
            continue
        seen.add(text)
        raw_id = value.get("id")
        comment_id = (
            str(raw_id)
            if raw_id
            else hashlib.sha256(f"{url}\0{text}".encode()).hexdigest()[:20]
        )
        empathy = value.get("empathy_count")
        try:
            empathy_count = int(empathy) if empathy is not None else None
        except (TypeError, ValueError):
            empathy_count = None
        comments.append(
            Comment(
                comment_id=comment_id,
                article_id=article_id,
                text=text,
                posted_at=_parse_datetime(value.get("posted_at")),
                order_index=len(comments),
                empathy_count=empathy_count,
            )
        )
    warnings = []
    if not comments:
        warnings.append(
            "公開記事HTMLにコメントが含まれていません。"
            "サイトがコメントページの自動取得を許可していない場合は本文のみ分析します。"
        )
    return WebIngestionResult(article=article, comments=comments, warnings=warnings)


class WebIngestionService:
    def __init__(
        self,
        settings: Settings,
        http_client: SafeHttpClient | None = None,
        yahoo_comment_provider: YahooCommentProvider | None = None,
    ) -> None:
        self.settings = settings
        self.http = http_client or SafeHttpClient(settings)
        self.yahoo_comments = yahoo_comment_provider
        if self.yahoo_comments is None and settings.yahoo_comment_fetch_enabled:
            self.yahoo_comments = SeleniumYahooCommentProvider(settings)

    def _robots_parser(self, url: str) -> RobotFileParser:
        robots_url = _robots_url(url)
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            robots_text, _ = self.http.get_text(robots_url, check_html=False)
        except WebIngestionError as exc:
            if "HTTP 404" not in str(exc):
                raise WebIngestionError("robots.txtを確認できないため取得を中止しました。") from exc
            robots_text = ""
        parser.parse(robots_text.splitlines())
        return parser

    def fetch(self, url: str, comment_limit: int | None = None) -> WebIngestionResult:
        normalized = normalize_public_url(url)
        parser = self._robots_parser(normalized)
        if not parser.can_fetch(self.settings.web_fetch_user_agent, normalized):
            raise WebIngestionError("このサイトのrobots.txtでは自動取得が許可されていません。")
        html, final_url = self.http.get_text(normalized)
        original_host = urlsplit(normalized).netloc
        if urlsplit(final_url).netloc != original_host:
            parser = self._robots_parser(final_url)
        if not parser.can_fetch(self.settings.web_fetch_user_agent, final_url):
            raise WebIngestionError("リダイレクト先の自動取得が許可されていません。")
        result = extract_public_page(html, final_url)
        host = (urlsplit(final_url).hostname or "").lower()
        if host == "news.yahoo.co.jp" and self.yahoo_comments is not None:
            try:
                comments = self.yahoo_comments.fetch(
                    result.article,
                    comment_limit or self.settings.analysis_max_comments,
                )
                warnings = [
                    warning
                    for warning in result.warnings
                    if not warning.startswith("公開記事HTMLにコメントが含まれていません")
                ]
                if not comments:
                    warnings.append("Yahooコメントは0件でした。")
                return WebIngestionResult(
                    article=result.article,
                    comments=comments,
                    warnings=warnings,
                )
            except YahooCommentFetchError as exc:
                return WebIngestionResult(
                    article=result.article,
                    comments=result.comments,
                    warnings=[*result.warnings, str(exc)],
                )
        return result
