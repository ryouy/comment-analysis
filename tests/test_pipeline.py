from pathlib import Path

from src.analysis.pipeline import AnalysisPipeline
from src.config.settings import Settings
from src.ingestion.normalized_loader import load_dataset
from src.llm.client import FakeOpenAIAnalysisClient
from src.storage.cache import FileCache


def test_full_pipeline_generates_all_features_and_uses_cache(tmp_path: Path) -> None:
    article, comments = load_dataset("data/sample_article.json")
    fake = FakeOpenAIAnalysisClient()
    settings = Settings(
        cache_dir=tmp_path,
        article_link_threshold=0.35,
        similarity_link_threshold=0.70,
        propagation_strong_threshold=0.80,
    )
    pipeline = AnalysisPipeline(
        settings=settings,
        llm_client=fake,
        cache=FileCache(tmp_path),
    )
    first = pipeline.run(article, comments)
    expected = {
        "emotion_timeline",
        "galaxy",
        "trigger_heatmap",
        "gap_index",
        "minority_signals",
        "propagation",
        "quality_frontier",
        "rhetoric_lens",
        "topic_drift",
        "semantic_cloud",
        "health",
        "framing",
    }
    assert first.run.status.value == "COMPLETED"
    assert expected == set(first.features)
    assert first.features["emotion_timeline"]["change_points"]
    calls = (fake.comment_calls, fake.framing_calls)

    second = pipeline.run(article, comments)
    assert second.cache_hit is True
    assert calls == (fake.comment_calls, fake.framing_calls)


def test_empty_comments_complete_without_api_failure(tmp_path: Path) -> None:
    article, _ = load_dataset("data/sample_article.json")
    result = AnalysisPipeline(
        settings=Settings(cache_dir=tmp_path),
        llm_client=FakeOpenAIAnalysisClient(),
    ).run(article, [])
    assert result.run.status.value == "COMPLETED"
    assert result.features["galaxy"]["available"] is False

