from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from src.analysis.derived import (
    build_clusters,
    build_framing,
    build_gap_index,
    build_health,
    build_minority_signals,
    build_propagation,
    build_quality,
    build_rhetoric,
    build_semantic_cloud,
    build_topic_drift,
    build_trigger_heatmap,
)
from src.analysis.emotion_seismograph import build_emotion_timeline
from src.analysis.galaxy import build_galaxy
from src.config.settings import Settings
from src.domain.models import (
    EMOTIONS,
    AnalysisBundle,
    AnalysisRun,
    Article,
    ArticleSentence,
    Comment,
    CommentAnalysis,
    RunStatus,
)
from src.embeddings.client import (
    EmbeddingClient,
    LocalEmbeddingClient,
    OpenAIEmbeddingClient,
)
from src.llm.client import (
    PROMPT_VERSION,
    LocalAnalysisClient,
    OpenAIAnalysisClient,
    TextAnalysisClient,
)
from src.llm.schemas import CommentBatchResponse, FramingResponse
from src.preprocessing.text import clean_text, split_sentences, tokenize
from src.storage.cache import FileCache, build_cache_key

PIPELINE_VERSION = "1.0.0"
ProgressCallback = Callable[[str, float], None]


class EmbeddingPayload(BaseModel):
    vectors: list[list[float]]


def _noop_progress(label: str, progress: float) -> None:
    del label, progress


class AnalysisPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: TextAnalysisClient | None = None,
        embedding_client: EmbeddingClient | None = None,
        cache: FileCache | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        if llm_client is not None:
            self.llm = llm_client
        elif self.settings.openai_api_key:
            self.llm = OpenAIAnalysisClient(self.settings)
        else:
            self.llm = LocalAnalysisClient()
        if embedding_client is not None:
            self.embedder = embedding_client
        elif self.settings.openai_api_key:
            self.embedder = OpenAIEmbeddingClient(self.settings)
        else:
            self.embedder = LocalEmbeddingClient()
        self.fallback_llm = LocalAnalysisClient()
        self.cache = cache or FileCache(self.settings.cache_dir)

    def run(
        self,
        article: Article,
        comments: list[Comment],
        *,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> AnalysisBundle:
        notify = progress or _noop_progress
        limited = sorted(comments, key=lambda item: item.order_index)[
            : self.settings.analysis_max_comments
        ]
        key = build_cache_key(
            article, limited, self.settings, PIPELINE_VERSION, PROMPT_VERSION
        )
        if not force:
            cached = self.cache.get("bundle", key, AnalysisBundle)
            if isinstance(cached, AnalysisBundle):
                return cached.model_copy(update={"cache_hit": True})

        started = datetime.now(UTC)
        run = AnalysisRun(
            run_id=str(uuid4()),
            article_id=article.article_id,
            status=RunStatus.INGESTED,
            started_at=started,
            pipeline_version=PIPELINE_VERSION,
            prompt_version=PROMPT_VERSION,
            embedding_model=self.settings.openai_embedding_model,
            text_model=self.settings.openai_text_model,
            comment_count=len(limited),
            config_snapshot=self.settings.snapshot(),
        )
        warnings = []
        if len(comments) > len(limited):
            warnings.append(
                f"設定上限により{len(comments)}件中{len(limited)}件を分析しました。"
            )
        if not comments:
            warnings.append("コメントがないため本文分析のみ表示します。")
        if any(comment.posted_at is None for comment in limited):
            warnings.append("投稿時刻がないため、感情推移は投稿順を使用します。")
        if not self.settings.openai_api_key:
            warnings.append("OpenAI未設定のため、ローカル特徴量による暫定分析です。")

        notify("本文を分割中", 0.08)
        sentence_values = [(-1, 0, article.title)] + split_sentences(article.body)
        sentences = [
            ArticleSentence(
                sentence_id=f"s{index:03d}",
                article_id=article.article_id,
                paragraph_index=paragraph,
                sentence_index=sentence,
                text=text,
                is_headline=index == 0,
            )
            for index, (paragraph, sentence, text) in enumerate(sentence_values)
        ]
        run.status = RunStatus.PREPROCESSED

        notify("コメントを前処理中", 0.18)
        cleaned = [clean_text(comment.text) for comment in limited]
        notify("意味ベクトルを生成中", 0.30)
        embedding_cache = None if force else self.cache.get(
            "embeddings", key, EmbeddingPayload
        )
        if isinstance(embedding_cache, EmbeddingPayload):
            all_vectors = embedding_cache.vectors
        else:
            all_texts = [sentence.text for sentence in sentences] + cleaned
            try:
                all_vectors = self.embedder.embed(all_texts)
            except Exception as exc:
                warnings.append(
                    "Embedding APIに失敗したためローカル特徴ハッシュを使用しました: "
                    f"{type(exc).__name__}"
                )
                all_vectors = LocalEmbeddingClient().embed(all_texts)
            self.cache.set("embeddings", key, EmbeddingPayload(vectors=all_vectors))
        sentence_vectors = all_vectors[: len(sentences)]
        embeddings = all_vectors[len(sentences) :]
        for sentence, vector in zip(sentences, sentence_vectors, strict=True):
            sentence.embedding = vector
        run.status = RunStatus.EMBEDDED

        notify("感情を分析中", 0.42)
        llm_cache = None if force else self.cache.get("llm", key, CommentBatchResponse)
        if isinstance(llm_cache, CommentBatchResponse):
            llm_items = llm_cache.items
        else:
            llm_items = []
            try:
                for start in range(0, len(limited), self.settings.comment_batch_size):
                    batch = limited[start : start + self.settings.comment_batch_size]
                    llm_items.extend(
                        self.llm.analyze_comments(article.title, article.body, batch).items
                    )
            except Exception as exc:
                warnings.append(
                    "OpenAI分析に失敗したためローカル暫定値を使用しました: "
                    f"{type(exc).__name__}"
                )
                llm_items = self.fallback_llm.analyze_comments(
                    article.title, article.body, limited
                ).items
            self.cache.set("llm", key, CommentBatchResponse(items=llm_items))
        llm_by_id = {item.comment_id: item for item in llm_items}
        analyses = []
        for comment, text, embedding in zip(limited, cleaned, embeddings, strict=True):
            item = llm_by_id.get(comment.comment_id)
            if item is None:
                item = self.fallback_llm.analyze_comments(
                    article.title, article.body, [comment]
                ).items[0]
            dominant = max(EMOTIONS, key=lambda name: item.emotion_scores[name])
            information_density = min(1.0, len(set(tokenize(text))) / 18)
            analyses.append(
                CommentAnalysis(
                    comment_id=comment.comment_id,
                    cleaned_text=text,
                    token_count=len(tokenize(text)),
                    embedding=embedding,
                    stance_label=item.stance_label,
                    stance_confidence=item.stance_confidence,
                    emotion_scores=item.emotion_scores,
                    dominant_emotion=dominant,
                    claim=item.claim,
                    reasons=item.reasons,
                    evidence_expressions=item.evidence_expressions,
                    target_entities=item.target_entities,
                    headline_dependency_score=item.headline_dependency_score,
                    specificity_score=item.specificity_score,
                    evidence_score=item.evidence_score,
                    logical_coherence_score=item.logical_coherence_score,
                    constructiveness_score=item.constructiveness_score,
                    respectfulness_score=item.respectfulness_score,
                    information_density_score=information_density,
                    rhetoric_flags=item.rhetoric_flags,
                    toxicity_probability=1 - item.respectfulness_score,
                    uncertainty_notes=item.uncertainty_notes,
                )
            )
        run.status = RunStatus.ENRICHED

        notify("意見クラスターを作成中", 0.55)
        galaxy = build_galaxy(limited, analyses, self.settings)
        trigger = build_trigger_heatmap(
            sentences, limited, analyses, self.settings.article_link_threshold
        )
        clusters = build_clusters(analyses)

        notify("議論指標を計算中", 0.72)
        propagation = build_propagation(limited, analyses, self.settings)
        framing_cache = None if force else self.cache.get("framing", key, FramingResponse)
        if isinstance(framing_cache, FramingResponse):
            framing_response = framing_cache
        else:
            try:
                framing_response = self.llm.analyze_framing(article.title, article.body)
            except Exception:
                framing_response = self.fallback_llm.analyze_framing(
                    article.title, article.body
                )
                warnings.append("見出し生成はローカル暫定値です。")
            self.cache.set("framing", key, framing_response)
        features: dict[str, Any] = {
            "emotion_timeline": build_emotion_timeline(
                limited, analyses, self.settings.change_point_sensitivity
            ),
            "galaxy": galaxy,
            "trigger_heatmap": trigger,
            "gap_index": build_gap_index(article, sentences, analyses),
            "minority_signals": build_minority_signals(
                limited, analyses, clusters, self.settings
            ),
            "propagation": propagation,
            "quality_frontier": build_quality(limited, analyses),
            "rhetoric_lens": build_rhetoric(analyses),
            "topic_drift": build_topic_drift(sentences, analyses),
            "semantic_cloud": build_semantic_cloud(limited, analyses),
            "health": build_health(analyses, clusters, propagation),
            "framing": build_framing(article, analyses, framing_response),
        }
        notify("可視化を作成中", 0.92)
        run.status = RunStatus.COMPLETED
        run.completed_at = datetime.now(UTC)
        bundle = AnalysisBundle(
            run=run,
            article=article,
            sentences=sentences,
            comments=limited,
            analyses=analyses,
            clusters=clusters,
            features=features,
            warnings=warnings,
        )
        self.cache.set("bundle", key, bundle)
        notify("完了", 1.0)
        return bundle
