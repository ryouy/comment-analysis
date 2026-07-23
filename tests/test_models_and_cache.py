from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.settings import Settings
from src.domain.models import EMOTIONS, Article, Comment, CommentAnalysis
from src.llm.schemas import CommentBatchResponse, FramingResponse
from src.storage.cache import build_cache_key


def test_comment_analysis_requires_all_emotions() -> None:
    with pytest.raises(ValidationError):
        CommentAnalysis(
            comment_id="c1",
            cleaned_text="test",
            emotion_scores={"anger": 0.2},
        )


def test_cache_key_is_order_independent_and_model_sensitive() -> None:
    article = Article(article_id="a", title="題", body="本文")
    comments = [
        Comment(comment_id="1", article_id="a", text="一", order_index=0),
        Comment(comment_id="2", article_id="a", text="二", order_index=1),
    ]
    settings = Settings(cache_dir=Path("/tmp/test-cache"))
    first = build_cache_key(article, comments, settings, "1", "1")
    second = build_cache_key(article, list(reversed(comments)), settings, "1", "1")
    changed = build_cache_key(
        article,
        comments,
        Settings(openai_text_model="different", cache_dir=Path("/tmp/test-cache")),
        "1",
        "1",
    )
    assert first == second
    assert first != changed


def test_default_emotions_are_valid() -> None:
    analysis = CommentAnalysis(comment_id="c", cleaned_text="x")
    assert set(analysis.emotion_scores) == set(EMOTIONS)


def test_openai_output_schemas_are_strict_and_fully_required() -> None:
    def validate(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                properties = value.get("properties")
                assert isinstance(properties, dict)
                assert value.get("additionalProperties") is False
                assert set(value.get("required", [])) == set(properties)
            for nested in value.values():
                validate(nested)
        elif isinstance(value, list):
            for nested in value:
                validate(nested)

    validate(CommentBatchResponse.model_json_schema())
    validate(FramingResponse.model_json_schema())
