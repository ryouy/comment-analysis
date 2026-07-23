from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = None
    openai_text_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_request_timeout_seconds: float = 60
    openai_max_retries: int = 3
    analysis_max_comments: int = 5000
    analysis_default_mode: str = "standard"
    comment_batch_size: int = 30
    embedding_batch_size: int = 256
    umap_random_state: int = 42
    hdbscan_min_cluster_size: int = 10
    similarity_link_threshold: float = 0.82
    propagation_strong_threshold: float = 0.88
    article_link_threshold: float = 0.45
    minority_cluster_share_max: float = 0.10
    rhetoric_display_threshold: float = 0.60
    change_point_sensitivity: str = "medium"
    cache_dir: Path = Path(".cache")
    store_raw_comments: bool = False
    anonymize_display_names: bool = True
    web_fetch_timeout_seconds: float = 15
    web_fetch_max_bytes: int = 5_000_000
    web_fetch_user_agent: str = "CommentAnalysis/1.0"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_text_model=os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini"),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            openai_request_timeout_seconds=float(
                os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "60")
            ),
            openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            analysis_max_comments=int(os.getenv("ANALYSIS_MAX_COMMENTS", "5000")),
            analysis_default_mode=os.getenv("ANALYSIS_DEFAULT_MODE", "standard"),
            comment_batch_size=int(os.getenv("COMMENT_BATCH_SIZE", "30")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "256")),
            umap_random_state=int(os.getenv("UMAP_RANDOM_STATE", "42")),
            hdbscan_min_cluster_size=int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "10")),
            similarity_link_threshold=float(
                os.getenv("SIMILARITY_LINK_THRESHOLD", "0.82")
            ),
            propagation_strong_threshold=float(
                os.getenv("PROPAGATION_STRONG_THRESHOLD", "0.88")
            ),
            article_link_threshold=float(os.getenv("ARTICLE_LINK_THRESHOLD", "0.45")),
            minority_cluster_share_max=float(
                os.getenv("MINORITY_CLUSTER_SHARE_MAX", "0.10")
            ),
            rhetoric_display_threshold=float(
                os.getenv("RHETORIC_DISPLAY_THRESHOLD", "0.60")
            ),
            change_point_sensitivity=os.getenv("CHANGE_POINT_SENSITIVITY", "medium"),
            cache_dir=Path(os.getenv("ANALYSIS_CACHE_DIR", ".cache")),
            store_raw_comments=_bool("STORE_RAW_COMMENTS", False),
            anonymize_display_names=_bool("ANONYMIZE_DISPLAY_NAMES", True),
            web_fetch_timeout_seconds=float(
                os.getenv("WEB_FETCH_TIMEOUT_SECONDS", "15")
            ),
            web_fetch_max_bytes=int(os.getenv("WEB_FETCH_MAX_BYTES", "5000000")),
            web_fetch_user_agent=os.getenv(
                "WEB_FETCH_USER_AGENT", "CommentAnalysis/1.0"
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        values = asdict(self)
        values["cache_dir"] = str(self.cache_dir)
        values["openai_api_key"] = bool(self.openai_api_key)
        return values
