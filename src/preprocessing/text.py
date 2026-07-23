from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter

URL_RE = re.compile(r"https?://\S+")
HTML_RE = re.compile(r"<[^>]+>")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
SENTENCE_RE = re.compile(r"(?<=[。！？!?])\s*|\n+")
TOKEN_RE = re.compile(r"[一-龥ぁ-んァ-ヶーA-Za-z0-9]{2,}")

DEFAULT_STOPWORDS = {
    "です",
    "ます",
    "思う",
    "感じる",
    "こと",
    "もの",
    "よう",
    "ため",
    "これ",
    "それ",
    "記事",
    "ニュース",
}


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(text))
    normalized = URL_RE.sub(" ", normalized)
    normalized = HTML_RE.sub(" ", normalized)
    normalized = CONTROL_RE.sub("", normalized)
    return re.sub(r"[ \t]+", " ", normalized).strip()


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    results: list[tuple[int, int, str]] = []
    for paragraph_index, paragraph in enumerate(text.splitlines()):
        if not paragraph.strip():
            continue
        for sentence_index, sentence in enumerate(SENTENCE_RE.split(paragraph)):
            value = clean_text(sentence)
            if value:
                results.append((paragraph_index, sentence_index, value))
    return results


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(clean_text(text))]


def extract_phrases(
    texts: list[str], stopwords: set[str] | None = None, limit: int = 40
) -> list[tuple[str, int]]:
    blocked = DEFAULT_STOPWORDS | (stopwords or set())
    phrases: Counter[str] = Counter()
    for text in texts:
        tokens = [token for token in tokenize(text) if token not in blocked]
        phrases.update(tokens)
        phrases.update("・".join(tokens[index : index + 2]) for index in range(len(tokens) - 1))
    return phrases.most_common(limit)

