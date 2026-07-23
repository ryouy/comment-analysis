from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.domain.models import EMOTIONS, RhetoricFlag


class LLMCommentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment_id: str
    stance_label: str | None = None
    stance_confidence: float = Field(default=0, ge=0, le=1)
    emotion_scores: dict[str, float]
    claim: str | None = None
    reasons: list[str] = Field(default_factory=list)
    evidence_expressions: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    headline_dependency_score: float = Field(default=0, ge=0, le=1)
    specificity_score: float = Field(default=0, ge=0, le=1)
    evidence_score: float = Field(default=0, ge=0, le=1)
    logical_coherence_score: float = Field(default=0, ge=0, le=1)
    constructiveness_score: float = Field(default=0, ge=0, le=1)
    respectfulness_score: float = Field(default=1, ge=0, le=1)
    rhetoric_flags: list[RhetoricFlag] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_emotions(self) -> LLMCommentItem:
        if set(self.emotion_scores) != set(EMOTIONS):
            raise ValueError("exactly nine emotion keys are required")
        if any(not 0 <= value <= 1 for value in self.emotion_scores.values()):
            raise ValueError("emotion values must be between 0 and 1")
        return self


class CommentBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[LLMCommentItem]


class FramingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emotional_terms: list[str] = Field(default_factory=list)
    emphasis_terms: list[str] = Field(default_factory=list)
    omitted_conditions: list[str] = Field(default_factory=list)
    body_alignment: float = Field(ge=0, le=1)
    neutral_headlines: dict[str, str] = Field(default_factory=dict)
    uncertainty_notes: list[str] = Field(default_factory=list)

