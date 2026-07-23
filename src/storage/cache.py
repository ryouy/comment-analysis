from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config.settings import Settings
from src.domain.models import Article, Comment


def build_cache_key(
    article: Article,
    comments: list[Comment],
    settings: Settings,
    pipeline_version: str,
    prompt_version: str,
) -> str:
    comment_values = sorted(
        "\0".join(
            (
                comment.comment_id,
                comment.text,
                comment.posted_at.isoformat() if comment.posted_at else "",
            )
        )
        for comment in comments
    )
    payload = {
        "title": article.title,
        "body": article.body,
        "comments": comment_values,
        "pipeline_version": pipeline_version,
        "prompt_version": prompt_version,
        "embedding_model": settings.openai_embedding_model,
        "text_model": settings.openai_text_model,
        "analysis_config": settings.snapshot(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class FileCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, namespace: str, key: str) -> Path:
        return self.directory / namespace / f"{key}.json"

    def get(self, namespace: str, key: str, model: type[BaseModel]) -> BaseModel | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def set(self, namespace: str, key: str, value: BaseModel | dict[str, Any]) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)
