from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.domain.models import Article, Comment


class NormalizedDataset(BaseModel):
    article: Article
    comments: list[Comment] = Field(default_factory=list)


def parse_dataset(data: dict[str, Any]) -> tuple[Article, list[Comment]]:
    parsed = NormalizedDataset.model_validate(data)
    article_id = parsed.article.article_id
    comments = [
        comment.model_copy(
            update={"article_id": article_id, "order_index": index}
            if comment.order_index < 0
            else {"article_id": article_id}
        )
        for index, comment in enumerate(parsed.comments)
    ]
    return parsed.article, comments


def load_dataset(path: str | Path) -> tuple[Article, list[Comment]]:
    with Path(path).open(encoding="utf-8") as stream:
        return parse_dataset(json.load(stream))
