from datetime import UTC, datetime, timedelta

from src.analysis.derived import build_gap_index, build_propagation
from src.analysis.emotion_seismograph import build_emotion_timeline
from src.analysis.vector import local_embedding
from src.config.settings import Settings
from src.domain.models import (
    EMOTIONS,
    Article,
    ArticleSentence,
    Comment,
    CommentAnalysis,
)


def _analysis(comment_id: str, anger: float, text: str) -> CommentAnalysis:
    emotions = {name: 0.05 for name in EMOTIONS}
    emotions["anger"] = anger
    return CommentAnalysis(
        comment_id=comment_id,
        cleaned_text=text,
        embedding=local_embedding(text),
        emotion_scores=emotions,
        dominant_emotion="anger",
        relevance_score=0.7,
    )


def test_emotion_change_point_detects_inserted_shift() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    comments = []
    analyses = []
    for index in range(40):
        angry = index >= 20
        text = "許せない 怒り 説明責任" if angry else "落ち着いた意見"
        comments.append(
            Comment(
                comment_id=f"c{index}",
                article_id="a",
                text=text,
                posted_at=start + timedelta(minutes=index),
                order_index=index,
            )
        )
        analyses.append(_analysis(f"c{index}", 0.95 if angry else 0.05, text))
    timeline = build_emotion_timeline(comments, analyses)
    assert timeline["axis_mode"] == "time"
    assert timeline["change_points"]
    assert "anger" in timeline["change_points"][0]["affected_emotions"]


def test_emotion_timeline_falls_back_to_order() -> None:
    comment = Comment(comment_id="c", article_id="a", text="意見", order_index=0)
    timeline = build_emotion_timeline([comment], [_analysis("c", 0.2, "意見")])
    assert timeline["axis_mode"] == "order"


def test_propagation_edges_never_go_backward_and_distinguish_exact() -> None:
    comments = [
        Comment(comment_id="a", article_id="x", text="予算の説明が必要", order_index=0),
        Comment(comment_id="b", article_id="x", text="予算の説明が必要", order_index=1),
    ]
    analyses = [_analysis(item.comment_id, 0.1, item.text) for item in comments]
    result = build_propagation(
        comments,
        analyses,
        Settings(similarity_link_threshold=0.1, propagation_strong_threshold=0.1),
    )
    assert result["edges"][0]["source"] == "a"
    assert result["edges"][0]["target"] == "b"
    assert result["edges"][0]["kind"] == "完全一致"


def test_gap_index_has_components_and_low_sample_warning() -> None:
    article = Article(article_id="a", title="見出し", body="本文です。")
    sentences = [
        ArticleSentence(
            sentence_id="s0",
            article_id="a",
            paragraph_index=-1,
            sentence_index=0,
            text="見出し",
            embedding=local_embedding("見出し"),
            is_headline=True,
        )
    ]
    result = build_gap_index(article, sentences, [_analysis("c", 0.1, "本文")])
    assert 0 <= result["value"] <= 100
    assert len(result["components"]) == 5
    assert result["warning"]

