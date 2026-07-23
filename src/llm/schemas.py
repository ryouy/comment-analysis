from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.domain.models import RhetoricFlag


class EmotionScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anger: float = Field(ge=0, le=1)
    anxiety: float = Field(ge=0, le=1)
    disappointment: float = Field(ge=0, le=1)
    ridicule: float = Field(ge=0, le=1)
    empathy: float = Field(ge=0, le=1)
    hope: float = Field(ge=0, le=1)
    doubt: float = Field(ge=0, le=1)
    resignation: float = Field(ge=0, le=1)
    moral_outrage: float = Field(ge=0, le=1)


class NeutralHeadlines(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_focused: str
    context_focused: str
    cautious: str


class LLMCommentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment_id: str
    stance_label: str | None
    stance_confidence: float = Field(ge=0, le=1)
    emotion_scores: EmotionScores
    claim: str | None
    reasons: list[str]
    evidence_expressions: list[str]
    target_entities: list[str]
    headline_dependency_score: float = Field(ge=0, le=1)
    specificity_score: float = Field(ge=0, le=1)
    evidence_score: float = Field(ge=0, le=1)
    logical_coherence_score: float = Field(ge=0, le=1)
    constructiveness_score: float = Field(ge=0, le=1)
    respectfulness_score: float = Field(ge=0, le=1)
    rhetoric_flags: list[RhetoricFlag]
    uncertainty_notes: list[str]


class CommentBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[LLMCommentItem]


class FramingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emotional_terms: list[str]
    emphasis_terms: list[str]
    omitted_conditions: list[str]
    body_alignment: float = Field(ge=0, le=1)
    neutral_headlines: NeutralHeadlines
    uncertainty_notes: list[str]
