from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

from src.domain.models import EMOTIONS, Comment, CommentAnalysis, EmotionChangePoint
from src.preprocessing.text import extract_phrases


def _window_size(count: int) -> int:
    if count < 100:
        return max(3, min(10, count))
    if count < 1000:
        return max(20, min(50, int(np.sqrt(count) * 2)))
    return max(50, min(100, count // 20))


def build_emotion_timeline(
    comments: list[Comment], analyses: list[CommentAnalysis], sensitivity: str = "medium"
) -> dict[str, Any]:
    if not comments:
        return {"axis_mode": "order", "bins": [], "change_points": [], "window_size": 0}
    ordered = sorted(
        zip(comments, analyses, strict=True),
        key=lambda pair: (
            pair[0].posted_at is None,
            pair[0].posted_at or datetime.max,
            pair[0].order_index,
        ),
    )
    window = _window_size(len(ordered))
    bins: list[dict[str, Any]] = []
    for start in range(0, len(ordered), window):
        chunk = ordered[start : start + window]
        values = np.asarray(
            [[analysis.emotion_scores[name] for name in EMOTIONS] for _, analysis in chunk],
            dtype=float,
        )
        weights = np.asarray([1 + (comment.empathy_count or 0) for comment, _ in chunk])
        weighted = np.average(values, axis=0, weights=weights)
        phrases = extract_phrases([comment.text for comment, _ in chunk], limit=5)
        bins.append(
            {
                "index": len(bins),
                "start_order": chunk[0][0].order_index,
                "end_order": chunk[-1][0].order_index,
                "start_timestamp": chunk[0][0].posted_at.isoformat()
                if chunk[0][0].posted_at
                else None,
                "timestamp": chunk[-1][0].posted_at.isoformat()
                if chunk[-1][0].posted_at
                else None,
                "comment_count": len(chunk),
                "mean": dict(zip(EMOTIONS, np.mean(values, axis=0).tolist(), strict=True)),
                "median": dict(zip(EMOTIONS, np.median(values, axis=0).tolist(), strict=True)),
                "weighted_mean": dict(zip(EMOTIONS, weighted.tolist(), strict=True)),
                "phrases": [{"phrase": phrase, "count": count} for phrase, count in phrases],
                "comment_ids": [comment.comment_id for comment, _ in chunk],
            }
        )
    matrix = np.asarray([[item["mean"][name] for name in EMOTIONS] for item in bins])
    changes: list[EmotionChangePoint] = []
    if len(matrix) > 1:
        deltas = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
        multipliers = {"low": 0.2, "medium": 0.5, "high": 0.9}
        threshold = max(
            0.12,
            float(np.mean(deltas) + multipliers.get(sensitivity, 1.0) * np.std(deltas)),
        )
        for position in np.flatnonzero(deltas >= threshold):
            before, after = matrix[position], matrix[position + 1]
            difference = after - before
            affected = [
                emotion
                for emotion, delta in zip(EMOTIONS, difference, strict=True)
                if abs(delta) >= max(0.08, float(np.max(abs(difference))) * 0.5)
            ]
            phrase_before = Counter(
                {item["phrase"]: item["count"] for item in bins[position]["phrases"]}
            )
            phrase_after = Counter(
                {item["phrase"]: item["count"] for item in bins[position + 1]["phrases"]}
            )
            triggers = [
                phrase
                for phrase, uplift in (phrase_after - phrase_before).most_common(3)
                if uplift > 0
            ]
            changes.append(
                EmotionChangePoint(
                    position=int(position + 1),
                    timestamp=datetime.fromisoformat(
                        bins[position + 1]["start_timestamp"]
                    )
                    if bins[position + 1]["start_timestamp"]
                    else None,
                    order_index=bins[position + 1]["start_order"],
                    magnitude=float(deltas[position]),
                    affected_emotions=affected,
                    before_vector=dict(zip(EMOTIONS, before.tolist(), strict=True)),
                    after_vector=dict(zip(EMOTIONS, after.tolist(), strict=True)),
                    candidate_triggers=triggers or ["顕著な新規フレーズなし"],
                    representative_comment_ids=bins[position + 1]["comment_ids"][:3],
                    confidence=min(0.95, 0.5 + float(deltas[position])),
                )
            )
    return {
        "axis_mode": "time" if all(comment.posted_at for comment in comments) else "order",
        "window_size": window,
        "bins": bins,
        "change_points": [item.model_dump(mode="json") for item in changes],
        "note": "急変と同時期に増えた要素であり、因果関係を示すものではありません。",
    }
