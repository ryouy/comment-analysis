from __future__ import annotations

import hashlib
import math
import re
import shutil
from collections.abc import Callable
from datetime import datetime
from functools import partial
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait

from src.config.settings import Settings
from src.domain.models import Article, Comment
from src.preprocessing.text import clean_text

DriverFactory = Callable[[], WebDriver]
JST = ZoneInfo("Asia/Tokyo")


class YahooCommentFetchError(RuntimeError):
    """Yahooコメント取得を継続できない場合のエラー。"""


class YahooCommentProvider(Protocol):
    def fetch(self, article: Article, limit: int) -> list[Comment]: ...


def _integer(text: str) -> int:
    matches = re.findall(r"[\d,]+", text)
    if not matches:
        return 0
    return int(matches[-1].replace(",", ""))


def _comment_id_from_params(params: str) -> str | None:
    match = re.search(r"(?:^|;)cmt_id:([^;]+)", params)
    return match.group(1) if match else None


def _parse_yahoo_datetime(text: str, reference: datetime | None) -> datetime | None:
    match = re.search(r"(\d{1,2})/(\d{1,2}).*?(\d{1,2}):(\d{2})", text)
    if not match:
        return None
    month, day, hour, minute = (int(value) for value in match.groups())
    now = datetime.now(JST)
    year = reference.astimezone(JST).year if reference else now.year
    try:
        value = datetime(year, month, day, hour, minute, tzinfo=JST)
    except ValueError:
        return None
    if reference is None and value > now.replace(microsecond=0):
        value = value.replace(year=year - 1)
    return value


def _comments_url(article_url: str, page: int) -> str:
    parsed = urlsplit(article_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/comments"):
        path = path[: -len("/comments")]
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"{path}/comments", f"page={page}", "")
    )


def _reply_count_exceeds(driver: WebDriver, previous: int) -> bool:
    return (
        len(
            driver.find_elements(
                By.CSS_SELECTOR,
                '[data-cl-params*="_cl_vmodule:rep;"]',
            )
        )
        > previous
    )


class SeleniumYahooCommentProvider:
    def __init__(
        self,
        settings: Settings,
        driver_factory: DriverFactory | None = None,
    ) -> None:
        self.settings = settings
        self.driver_factory = driver_factory or self._create_driver

    def _create_driver(self) -> WebDriver:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1400,2000")
        options.add_argument("--lang=ja-JP")
        options.add_argument("--disable-extensions")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument(f"--user-agent={self.settings.web_fetch_user_agent}")
        chrome_binary = (
            self.settings.selenium_chrome_binary
            or shutil.which("chromium")
            or shutil.which("chromium-browser")
            or shutil.which("google-chrome")
        )
        if chrome_binary:
            options.binary_location = chrome_binary
        driver_path = self.settings.selenium_driver_path or shutil.which("chromedriver")
        service = (
            Service(executable_path=driver_path)
            if driver_path
            else Service()
        )
        try:
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(self.settings.selenium_page_timeout_seconds)
            return driver
        except WebDriverException as exc:
            raise YahooCommentFetchError(
                "Chrome/Chromiumまたは対応するDriverを起動できません。"
            ) from exc

    @staticmethod
    def _click_more(driver: WebDriver) -> None:
        elements = driver.find_elements(
            By.XPATH,
            "//*[self::button or self::a][contains(normalize-space(.), 'もっと見る')]",
        )
        for element in elements:
            try:
                driver.execute_script("arguments[0].click()", element)
            except (StaleElementReferenceException, WebDriverException):
                continue

    @staticmethod
    def _element_by_params(
        element: WebElement, module: str, link: str
    ) -> WebElement | None:
        selector = (
            f'[data-cl-params*="_cl_vmodule:{module};"]'
            f'[data-cl-params*="_cl_link:{link};"]'
        )
        matches = element.find_elements(By.CSS_SELECTOR, selector)
        return matches[0] if matches else None

    def _general_comment(
        self,
        wrapper: WebElement,
        article: Article,
        order_index: int,
    ) -> tuple[Comment | None, int]:
        time_link = self._element_by_params(wrapper, "cmt_usr", "prmtime")
        if time_link is None:
            return None, 0
        try:
            comment_element = time_link.find_element(By.XPATH, "ancestor::article[1]")
            paragraphs = comment_element.find_elements(By.CSS_SELECTOR, "p")
            text = clean_text("\n".join(item.text for item in paragraphs if item.text))
            if not text:
                return None, 0
            href = time_link.get_attribute("href") or ""
            comment_id = href.rstrip("/").rsplit("/", 1)[-1]
            if not comment_id:
                comment_id = hashlib.sha256(text.encode()).hexdigest()[:20]
            empathy_button = self._element_by_params(
                comment_element, "cmt_usr", "agbtn1"
            )
            reply_button = self._element_by_params(comment_element, "cmt_usr", "opnre")
            empathy_count = _integer(empathy_button.text) if empathy_button else None
            reply_count = _integer(reply_button.text) if reply_button else 0
            return (
                Comment(
                    comment_id=comment_id,
                    article_id=article.article_id,
                    text=text,
                    posted_at=_parse_yahoo_datetime(
                        time_link.text, article.published_at
                    ),
                    order_index=order_index,
                    empathy_count=empathy_count,
                    reply_count=reply_count,
                ),
                reply_count,
            )
        except (NoSuchElementException, StaleElementReferenceException):
            return None, 0

    def _expert_comments(
        self, driver: WebDriver, article: Article, start_index: int
    ) -> list[Comment]:
        comments: list[Comment] = []
        anchors = driver.find_elements(
            By.CSS_SELECTOR,
            '[data-cl-params*="_cl_vmodule:cmt_athr;"]'
            '[data-cl-params*="_cl_link:profnm;"]',
        )
        seen: set[str] = set()
        for anchor in anchors:
            try:
                element = anchor.find_element(By.XPATH, "ancestor::article[1]")
                paragraphs = element.find_elements(By.CSS_SELECTOR, "p")
                text = clean_text("\n".join(item.text for item in paragraphs if item.text))
                if not text or text in seen:
                    continue
                seen.add(text)
                time_elements = element.find_elements(By.CSS_SELECTOR, "time")
                reference_button = self._element_by_params(
                    element, "cmt_athr", "ref"
                )
                comment_id = "expert-" + hashlib.sha256(text.encode()).hexdigest()[:16]
                comments.append(
                    Comment(
                        comment_id=comment_id,
                        article_id=article.article_id,
                        text=text,
                        posted_at=_parse_yahoo_datetime(
                            time_elements[0].text if time_elements else "",
                            article.published_at,
                        ),
                        order_index=start_index + len(comments),
                        empathy_count=_integer(reference_button.text)
                        if reference_button
                        else None,
                    )
                )
            except (NoSuchElementException, StaleElementReferenceException):
                continue
        return comments

    def _reply_comments(
        self,
        driver: WebDriver,
        parent: Comment,
        start_index: int,
    ) -> list[Comment]:
        anchors = driver.find_elements(
            By.CSS_SELECTOR,
            '[data-cl-params*="_cl_vmodule:rep;"]'
            '[data-cl-params*="_cl_link:profnm;"]',
        )
        comments: list[Comment] = []
        seen: set[str] = set()
        for anchor in anchors:
            try:
                element = anchor.find_element(By.XPATH, "ancestor::article[1]")
                paragraphs = element.find_elements(By.CSS_SELECTOR, "p")
                text = clean_text("\n".join(item.text for item in paragraphs if item.text))
                if not text or text in seen:
                    continue
                seen.add(text)
                time_elements = element.find_elements(By.CSS_SELECTOR, "time")
                empathy_button = self._element_by_params(element, "rep", "agbtn1")
                params = empathy_button.get_attribute("data-cl-params") if empathy_button else ""
                comment_id = _comment_id_from_params(params or "")
                if not comment_id:
                    comment_id = hashlib.sha256(
                        f"{parent.comment_id}\0{text}".encode()
                    ).hexdigest()[:20]
                comments.append(
                    Comment(
                        comment_id=comment_id,
                        article_id=parent.article_id,
                        text=text,
                        posted_at=_parse_yahoo_datetime(
                            time_elements[0].text if time_elements else "",
                            parent.posted_at,
                        ),
                        order_index=start_index + len(comments),
                        empathy_count=_integer(empathy_button.text)
                        if empathy_button
                        else None,
                        parent_comment_id=parent.comment_id,
                    )
                )
            except (NoSuchElementException, StaleElementReferenceException):
                continue
        return comments

    def _expand_replies(
        self, driver: WebDriver, wrapper: WebElement, reply_count: int
    ) -> None:
        if reply_count <= 0:
            return
        button = self._element_by_params(wrapper, "cmt_usr", "opnre")
        if button is None:
            return
        before = len(
            driver.find_elements(
                By.CSS_SELECTOR, '[data-cl-params*="_cl_vmodule:rep;"]'
            )
        )
        try:
            driver.execute_script("arguments[0].click()", button)
            WebDriverWait(driver, 5).until(
                partial(_reply_count_exceeds, previous=before)
            )
        except (TimeoutException, StaleElementReferenceException, WebDriverException):
            return
        for _ in range(max(0, math.ceil(reply_count / 10) - 1)):
            more = driver.find_elements(
                By.XPATH,
                "//*[self::button or self::a]"
                "[contains(normalize-space(.), '返信をもっと見る')]",
            )
            if not more:
                break
            try:
                before = len(
                    driver.find_elements(
                        By.CSS_SELECTOR,
                        '[data-cl-params*="_cl_vmodule:rep;"]',
                    )
                )
                driver.execute_script("arguments[0].click()", more[0])
                WebDriverWait(driver, 3).until(
                    partial(_reply_count_exceeds, previous=before)
                )
            except (
                StaleElementReferenceException,
                TimeoutException,
                WebDriverException,
            ):
                break

    def fetch(self, article: Article, limit: int) -> list[Comment]:
        if not article.source_url:
            return []
        host = (urlsplit(article.source_url).hostname or "").lower()
        if host != "news.yahoo.co.jp":
            return []
        maximum = max(1, min(limit, self.settings.analysis_max_comments))
        pages = min(
            self.settings.yahoo_comment_fetch_max_pages,
            max(1, math.ceil(maximum / 10)),
        )
        comments: list[Comment] = []
        seen_ids: set[str] = set()
        driver = self.driver_factory()
        try:
            for page in range(1, pages + 1):
                try:
                    driver.get(_comments_url(article.source_url, page))
                    WebDriverWait(
                        driver, self.settings.selenium_page_timeout_seconds
                    ).until(
                        lambda current: current.find_elements(
                            By.CSS_SELECTOR,
                            'div[id^="viewable_comment_middle_"], #comment-main',
                        )
                    )
                except (TimeoutException, WebDriverException) as exc:
                    if page == 1:
                        raise YahooCommentFetchError(
                            "Yahooコメントページを読み込めませんでした。"
                        ) from exc
                    break
                self._click_more(driver)
                if page == 1:
                    for expert in self._expert_comments(
                        driver, article, len(comments)
                    ):
                        if expert.comment_id not in seen_ids and len(comments) < maximum:
                            seen_ids.add(expert.comment_id)
                            comments.append(expert)
                markers = driver.find_elements(
                    By.CSS_SELECTOR, 'div[id^="viewable_comment_middle_"]'
                )
                added_on_page = 0
                for marker in markers:
                    if len(comments) >= maximum:
                        break
                    try:
                        wrapper = marker.find_element(By.XPATH, "..")
                    except (NoSuchElementException, StaleElementReferenceException):
                        continue
                    parent, reply_count = self._general_comment(
                        wrapper, article, len(comments)
                    )
                    if parent is None or parent.comment_id in seen_ids:
                        continue
                    seen_ids.add(parent.comment_id)
                    comments.append(parent)
                    added_on_page += 1
                    if (
                        self.settings.yahoo_comment_fetch_replies
                        and reply_count
                        and len(comments) < maximum
                    ):
                        self._expand_replies(driver, wrapper, reply_count)
                        for reply in self._reply_comments(
                            driver, parent, len(comments)
                        ):
                            if (
                                reply.comment_id not in seen_ids
                                and len(comments) < maximum
                            ):
                                seen_ids.add(reply.comment_id)
                                comments.append(reply)
                if added_on_page == 0:
                    break
        finally:
            driver.quit()
        return comments
