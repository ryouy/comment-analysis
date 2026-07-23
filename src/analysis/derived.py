from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from src.analysis.vector import cosine, lexical_jaccard
from src.config.settings import Settings
from src.domain.models import (
    Article,
    ArticleSentence,
    ClusterSummary,
    Comment,
    CommentAnalysis,
    SentenceLink,
)
from src.llm.schemas import FramingResponse
from src.preprocessing.text import extract_phrases, tokenize


def build_clusters(analyses: list[CommentAnalysis]) -> list[ClusterSummary]:
    groups: defaultdict[str, list[CommentAnalysis]] = defaultdict(list)
    for analysis in analyses:
        groups[analysis.cluster_id or "unclustered"].append(analysis)
    result = []
    for cluster_id, items in sorted(groups.items()):
        emotions = {
            emotion: float(np.mean([item.emotion_scores[emotion] for item in items]))
            for emotion in items[0].emotion_scores
        }
        representative = sorted(
            items,
            key=lambda item: item.relevance_score + item.information_density_score,
            reverse=True,
        )[:3]
        result.append(
            ClusterSummary(
                cluster_id=cluster_id,
                label="ノイズ・孤立意見"
                if cluster_id == "noise"
                else cluster_id.replace("cluster-", "意見群 "),
                description="意味的に近いコメントの集合",
                size=len(items),
                share=len(items) / max(1, len(analyses)),
                representative_comment_ids=[item.comment_id for item in representative],
                central_claims=[item.claim for item in representative if item.claim],
                common_reasons=[
                    reason for item in representative for reason in item.reasons[:1]
                ],
                dominant_emotions=emotions,
                stance_label=next((item.stance_label for item in items if item.stance_label), None),
                novelty_score=float(np.mean([item.originality_score for item in items])),
            )
        )
    return result


def build_trigger_heatmap(
    sentences: list[ArticleSentence],
    comments: list[Comment],
    analyses: list[CommentAnalysis],
    threshold: float,
) -> dict[str, Any]:
    rows = []
    outside = []
    by_sentence: defaultdict[str, list[CommentAnalysis]] = defaultdict(list)
    for comment, analysis in zip(comments, analyses, strict=True):
        similarities = sorted(
            (
                (sentence.sentence_id, cosine(analysis.embedding or [], sentence.embedding or []))
                for sentence in sentences
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        selected = [pair for pair in similarities[:3] if pair[1] >= threshold]
        if not selected:
            outside.append(comment.comment_id)
            analysis.relevance_score = similarities[0][1] if similarities else 0
            continue
        total = sum(score for _, score in selected)
        analysis.article_sentence_links = [
            SentenceLink(
                sentence_id=sentence_id,
                similarity=score,
                weight=score / total,
            )
            for sentence_id, score in selected
        ]
        analysis.relevance_score = selected[0][1]
        for sentence_id, _ in selected:
            by_sentence[sentence_id].append(analysis)
    for sentence in sentences:
        linked = by_sentence[sentence.sentence_id]
        rows.append(
            {
                "sentence_id": sentence.sentence_id,
                "text": sentence.text,
                "is_headline": sentence.is_headline,
                "paragraph_index": sentence.paragraph_index,
                "comment_count": len(linked),
                "comment_rate": len(linked) / max(1, len(comments)),
                "comment_ids": [item.comment_id for item in linked],
                "emotions": {
                    emotion: float(np.mean([item.emotion_scores[emotion] for item in linked]))
                    if linked
                    else 0
                    for emotion in analyses[0].emotion_scores
                }
                if analyses
                else {},
                "stance_diversity": len({item.stance_label for item in linked if item.stance_label})
                / max(1, len(linked)),
            }
        )
    return {"sentences": rows, "outside_comment_ids": outside}


def build_gap_index(
    article: Article,
    sentences: list[ArticleSentence],
    analyses: list[CommentAnalysis],
) -> dict[str, Any]:
    if not analyses:
        components = {
            "semantic_misalignment": 0.0,
            "external_topic_rate": 0.0,
            "headline_dependency_rate": 0.0,
            "interpretation_conflict_rate": 0.0,
            "body_coverage_gap": 1.0,
        }
    else:
        headline = sentences[0] if sentences else None
        linked_sentence_ids = {
            link.sentence_id for item in analyses for link in item.article_sentence_links
        }
        body_sentence_ids = {item.sentence_id for item in sentences if not item.is_headline}
        headline_rates = []
        for item in analyses:
            headline_similarity = (
                cosine(item.embedding or [], headline.embedding or []) if headline else 0
            )
            body_similarity = max(
                (
                    cosine(item.embedding or [], sentence.embedding or [])
                    for sentence in sentences
                    if not sentence.is_headline
                ),
                default=0,
            )
            item.headline_dependency_score = max(0.0, headline_similarity - body_similarity + 0.25)
            headline_rates.append(item.headline_dependency_score >= 0.55)
        components = {
            "semantic_misalignment": 1
            - float(np.mean([item.relevance_score for item in analyses])),
            "external_topic_rate": sum(not item.article_sentence_links for item in analyses)
            / len(analyses),
            "headline_dependency_rate": sum(headline_rates) / len(analyses),
            "interpretation_conflict_rate": sum(
                item.stance_label == "conflict" for item in analyses
            )
            / len(analyses),
            "body_coverage_gap": 1
            - len(linked_sentence_ids & body_sentence_ids) / max(1, len(body_sentence_ids)),
        }
    weights = {
        "semantic_misalignment": 0.30,
        "external_topic_rate": 0.25,
        "headline_dependency_rate": 0.20,
        "interpretation_conflict_rate": 0.15,
        "body_coverage_gap": 0.10,
    }
    value = 100 * sum(components[key] * weights[key] for key in weights)
    return {
        "value": round(value, 1),
        "components": components,
        "weights": weights,
        "confidence": min(1.0, len(analyses) / 30),
        "warning": "10コメント未満のため推定の不確実性が高いです"
        if len(analyses) < 10
        else None,
        "headline": article.title,
    }


def build_minority_signals(
    comments: list[Comment],
    analyses: list[CommentAnalysis],
    clusters: list[ClusterSummary],
    settings: Settings,
) -> list[dict[str, Any]]:
    shares = {cluster.cluster_id: cluster.share for cluster in clusters}
    results: list[dict[str, Any]] = []
    for comment, item in zip(comments, analyses, strict=True):
        if shares.get(item.cluster_id or "", 1) > settings.minority_cluster_share_max:
            continue
        blindspot = min(1.0, len(item.article_sentence_links) * 0.3)
        future = 0.6 if any(word in item.cleaned_text for word in ("将来", "今後", "影響")) else 0.2
        attention = math.log1p(comment.empathy_count or 0) / 8
        safety_penalty = max(0.0, 0.7 - item.respectfulness_score) * 0.5
        score = (
            0.25 * item.originality_score
            + 0.20 * item.specificity_score
            + 0.15 * item.evidence_score
            + 0.15 * blindspot
            + 0.10 * future
            + 0.10 * item.constructiveness_score
            + 0.05 * min(1.0, attention)
            - safety_penalty
        )
        if item.relevance_score < 0.35 or score < 0.25:
            continue
        results.append(
            {
                "comment_id": comment.comment_id,
                "score": score,
                "cluster_id": item.cluster_id,
                "cluster_share": shares.get(item.cluster_id or "", 0),
                "originality": item.originality_score,
                "specificity": item.specificity_score,
                "evidence": item.evidence_score,
                "blindspot_coverage": blindspot,
                "related_sentence_ids": [
                    link.sentence_id for link in item.article_sentence_links
                ],
            }
        )
    return sorted(results, key=lambda result: float(result["score"]), reverse=True)[:20]


def build_propagation(
    comments: list[Comment], analyses: list[CommentAnalysis], settings: Settings
) -> dict[str, Any]:
    nodes = [
        {
            "comment_id": comment.comment_id,
            "order_index": comment.order_index,
            "empathy_count": comment.empathy_count or 0,
            "cluster_id": analysis.cluster_id,
        }
        for comment, analysis in zip(comments, analyses, strict=True)
    ]
    edges = []
    for later_index, (later, later_analysis) in enumerate(
        zip(comments, analyses, strict=True)
    ):
        candidates = []
        for earlier_index in range(later_index):
            earlier, earlier_analysis = comments[earlier_index], analyses[earlier_index]
            semantic = cosine(
                earlier_analysis.embedding or [], later_analysis.embedding or []
            )
            lexical = lexical_jaccard(earlier.text, later.text)
            if semantic >= settings.propagation_strong_threshold or (
                semantic >= settings.similarity_link_threshold and lexical >= 0.35
            ):
                candidates.append((semantic + lexical * 0.2, earlier, semantic, lexical))
        for _, earlier, semantic, lexical in sorted(candidates, reverse=True)[:1]:
            exact = earlier.text.strip() == later.text.strip()
            kind = "完全一致" if exact else "部分一致" if lexical >= 0.6 else "意味的言い換え"
            edges.append(
                {
                    "source": earlier.comment_id,
                    "target": later.comment_id,
                    "similarity": semantic,
                    "lexical_jaccard": lexical,
                    "kind": kind,
                }
            )
    return {
        "nodes": nodes,
        "edges": edges,
        "duplicate_rate": sum(edge["kind"] == "完全一致" for edge in edges)
        / max(1, len(comments)),
        "paraphrase_rate": sum(edge["kind"] == "意味的言い換え" for edge in edges)
        / max(1, len(comments)),
        "note": "線は意味的類似性と投稿順を示し、コピーや影響関係を断定しません。",
    }


def build_quality(comments: list[Comment], analyses: list[CommentAnalysis]) -> list[dict[str, Any]]:
    return [
        {
            "comment_id": comment.comment_id,
            "cluster_id": item.cluster_id,
            "empathy_count": comment.empathy_count or 0,
            "specificity": item.specificity_score,
            "evidence": item.evidence_score,
            "originality": item.originality_score,
            "logical_coherence": item.logical_coherence_score,
            "relevance": item.relevance_score,
            "constructiveness": item.constructiveness_score,
            "respectfulness": item.respectfulness_score,
            "information_density": item.information_density_score,
            "insufficient_material": len(item.cleaned_text) < 20,
        }
        for comment, item in zip(comments, analyses, strict=True)
    ]


def build_rhetoric(analyses: list[CommentAnalysis]) -> dict[str, Any]:
    items = [
        {"comment_id": analysis.comment_id, **flag.model_dump()}
        for analysis in analyses
        for flag in analysis.rhetoric_flags
    ]
    counts = Counter(item["label"] for item in items)
    return {
        "items": items,
        "summary": [
            {"label": label, "count": count, "rate": count / max(1, len(analyses))}
            for label, count in counts.most_common()
        ],
    }


def build_topic_drift(
    sentences: list[ArticleSentence], analyses: list[CommentAnalysis]
) -> dict[str, Any]:
    links: Counter[tuple[str, str, str]] = Counter()
    evidence: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    for item in analyses:
        source = (
            item.article_sentence_links[0].sentence_id
            if item.article_sentence_links
            else "本文外"
        )
        cluster = item.cluster_id or "判断不能"
        category = (
            "直接議論"
            if item.relevance_score >= 0.65
            else "関連展開"
            if item.relevance_score >= 0.45
            else "本文外展開"
        )
        key = (source, cluster, category)
        links[key] += 1
        evidence[key].append(item.comment_id)
    return {
        "nodes": [
            {"id": sentence.sentence_id, "label": sentence.text[:35]} for sentence in sentences
        ]
        + [{"id": "本文外", "label": "本文外"}],
        "links": [
            {
                "source": source,
                "target": cluster,
                "category": category,
                "value": count,
                "comment_ids": evidence[(source, cluster, category)][:20],
            }
            for (source, cluster, category), count in links.items()
        ],
    }


def build_semantic_cloud(
    comments: list[Comment], analyses: list[CommentAnalysis]
) -> list[dict[str, Any]]:
    phrases = extract_phrases([comment.text for comment in comments], limit=50)
    result = []
    for phrase, count in phrases:
        matching = [
            item
            for comment, item in zip(comments, analyses, strict=True)
            if phrase.replace("・", "") in comment.text.replace("・", "")
            or all(token in comment.text for token in phrase.split("・"))
        ]
        emotion = (
            max(
                analyses[0].emotion_scores,
                key=lambda name: float(
                    np.mean([item.emotion_scores[name] for item in matching])
                ),
            )
            if matching and analyses
            else None
        )
        score = min(1.0, math.log1p(count) / 4 + len(matching) / max(1, len(comments)))
        result.append(
            {
                "phrase": phrase,
                "count": count,
                "score": score,
                "dominant_emotion": emotion,
                "comment_ids": [item.comment_id for item in matching[:20]],
                "sentence_ids": sorted(
                    {
                        link.sentence_id
                        for item in matching
                        for link in item.article_sentence_links
                    }
                ),
            }
        )
    return result


def build_health(
    analyses: list[CommentAnalysis],
    clusters: list[ClusterSummary],
    propagation: dict[str, Any],
) -> dict[str, Any]:
    if not analyses:
        values = {name: 0.0 for name in (
            "多様性", "分断度", "本文関連性", "根拠提示率", "建設性",
            "重複・同調度", "敬意・非攻撃性", "少数意見可視性", "橋渡し率",
        )}
    else:
        shares = np.asarray([cluster.share for cluster in clusters if cluster.share > 0])
        entropy = -float(np.sum(shares * np.log(shares))) / max(1e-9, math.log(max(2, len(shares))))
        minority = sum(cluster.share for cluster in clusters if cluster.share <= 0.1)
        values = {
            "多様性": 100 * entropy,
            "分断度": 100 * max(0.0, 1 - minority - entropy * 0.25),
            "本文関連性": 100 * float(np.mean([item.relevance_score for item in analyses])),
            "根拠提示率": 100 * float(np.mean([item.evidence_score for item in analyses])),
            "建設性": 100 * float(
                np.mean([item.constructiveness_score for item in analyses])
            ),
            "重複・同調度": 100
            * min(1.0, propagation["duplicate_rate"] + propagation["paraphrase_rate"]),
            "敬意・非攻撃性": 100
            * float(np.mean([item.respectfulness_score for item in analyses])),
            "少数意見可視性": 100 * minority,
            "橋渡し率": 100
            * sum(0.35 <= item.originality_score <= 0.65 for item in analyses)
            / len(analyses),
        }
    return {
        "metrics": values,
        "confidence": min(1.0, len(analyses) / 30),
        "warning": "取得できたコメント範囲の構造であり、社会全体の世論ではありません。",
    }


def build_framing(
    article: Article,
    analyses: list[CommentAnalysis],
    framing: FramingResponse,
) -> dict[str, Any]:
    title_tokens = set(tokenize(article.title))
    dependent = [
        item.comment_id for item in analyses if item.headline_dependency_score >= 0.55
    ]
    repeated = [
        item.comment_id
        for item in analyses
        if title_tokens & set(tokenize(item.cleaned_text))
    ]
    return {
        **framing.model_dump(mode="json"),
        "original_title": article.title,
        "headline_repetition_rate": len(repeated) / max(1, len(analyses)),
        "headline_dependency_rate": len(dependent) / max(1, len(analyses)),
        "repeated_comment_ids": repeated[:30],
        "dependent_comment_ids": dependent[:30],
        "note": "見出しの編集意図を断定せず、AIによる比較用分析として表示しています。",
    }
