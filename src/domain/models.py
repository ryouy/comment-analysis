from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

EMOTIONS = (
    "anger",
    "anxiety",
    "disappointment",
    "ridicule",
    "empathy",
    "hope",
    "doubt",
    "resignation",
    "moral_outrage",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Article(StrictModel):
    article_id: str
    source_url: str | None = None
    source_name: str | None = None
    title: str
    summary: str | None = None
    body: str
    published_at: datetime | None = None
    fetched_at: datetime | None = None
    category: str | None = None


class ArticleSentence(StrictModel):
    sentence_id: str
    article_id: str
    paragraph_index: int
    sentence_index: int
    text: str
    embedding: list[float] | None = None
    is_headline: bool = False


class Comment(StrictModel):
    comment_id: str
    article_id: str
    text: str
    posted_at: datetime | None = None
    order_index: int
    empathy_count: int | None = None
    reply_count: int | None = None
    parent_comment_id: str | None = None


class SentenceLink(StrictModel):
    sentence_id: str
    similarity: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)


class RhetoricFlag(StrictModel):
    label: str
    probability: float = Field(ge=0, le=1)
    evidence_span: str | None = None
    explanation: str
    uncertainty: str | None = None
    severity: str = "medium"


class CommentAnalysis(StrictModel):
    comment_id: str
    cleaned_text: str
    language: str = "ja"
    token_count: int = 0
    embedding: list[float] | None = None
    cluster_id: str | None = None
    stance_label: str | None = None
    stance_confidence: float | None = Field(default=None, ge=0, le=1)
    emotion_scores: dict[str, float] = Field(
        default_factory=lambda: {emotion: 0.0 for emotion in EMOTIONS}
    )
    dominant_emotion: str | None = None
    claim: str | None = None
    reasons: list[str] = Field(default_factory=list)
    evidence_expressions: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    article_sentence_links: list[SentenceLink] = Field(default_factory=list)
    headline_dependency_score: float = Field(default=0, ge=0, le=1)
    specificity_score: float = Field(default=0, ge=0, le=1)
    evidence_score: float = Field(default=0, ge=0, le=1)
    originality_score: float = Field(default=0, ge=0, le=1)
    logical_coherence_score: float = Field(default=0, ge=0, le=1)
    relevance_score: float = Field(default=0, ge=0, le=1)
    constructiveness_score: float = Field(default=0, ge=0, le=1)
    respectfulness_score: float = Field(default=1, ge=0, le=1)
    information_density_score: float = Field(default=0, ge=0, le=1)
    rhetoric_flags: list[RhetoricFlag] = Field(default_factory=list)
    toxicity_probability: float | None = Field(default=None, ge=0, le=1)
    uncertainty_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_emotions(self) -> CommentAnalysis:
        missing = set(EMOTIONS) - self.emotion_scores.keys()
        if missing:
            raise ValueError(f"emotion_scores missing: {sorted(missing)}")
        if any(not 0 <= value <= 1 for value in self.emotion_scores.values()):
            raise ValueError("emotion_scores must be between 0 and 1")
        return self


class ClusterSummary(StrictModel):
    cluster_id: str
    label: str
    description: str
    size: int
    share: float
    representative_comment_ids: list[str]
    central_claims: list[str]
    common_reasons: list[str]
    dominant_emotions: dict[str, float]
    stance_label: str | None = None
    novelty_score: float = 0


class RunStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    INGESTED = "INGESTED"
    PREPROCESSED = "PREPROCESSED"
    EMBEDDED = "EMBEDDED"
    CLUSTERED = "CLUSTERED"
    ENRICHED = "ENRICHED"
    VISUALIZED = "VISUALIZED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AnalysisRun(StrictModel):
    run_id: str
    article_id: str
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None = None
    pipeline_version: str
    prompt_version: str
    embedding_model: str
    text_model: str
    comment_count: int
    config_snapshot: dict[str, Any]
    error_message: str | None = None


class EmotionChangePoint(StrictModel):
    position: int
    timestamp: datetime | None
    order_index: int
    magnitude: float
    affected_emotions: list[str]
    before_vector: dict[str, float]
    after_vector: dict[str, float]
    candidate_triggers: list[str]
    representative_comment_ids: list[str]
    confidence: float


class AnalysisBundle(StrictModel):
    run: AnalysisRun
    article: Article
    sentences: list[ArticleSentence]
    comments: list[Comment]
    analyses: list[CommentAnalysis]
    clusters: list[ClusterSummary]
    features: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    cache_hit: bool = False
