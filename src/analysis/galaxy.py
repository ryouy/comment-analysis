from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from src.config.settings import Settings
from src.domain.models import Comment, CommentAnalysis


def _cluster(vectors: np.ndarray, settings: Settings) -> tuple[np.ndarray, str]:
    count = len(vectors)
    try:
        if count >= max(30, settings.hdbscan_min_cluster_size * 2):
            import hdbscan

            labels = hdbscan.HDBSCAN(
                min_cluster_size=settings.hdbscan_min_cluster_size,
                metric="euclidean",
            ).fit_predict(vectors)
            if any(label >= 0 for label in labels):
                return labels, "HDBSCAN"
    except ImportError:
        pass
    best_labels = np.zeros(count, dtype=int)
    best_score = -1.0
    for clusters in range(2, min(8, count - 1) + 1):
        with threadpool_limits(limits=1):
            labels = KMeans(
                n_clusters=clusters, random_state=42, n_init=3
            ).fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(vectors, labels))
        if score > best_score:
            best_score, best_labels = score, labels
    return best_labels, "KMeans fallback"


def build_galaxy(
    comments: list[Comment], analyses: list[CommentAnalysis], settings: Settings
) -> dict[str, Any]:
    if len(comments) < 5:
        return {"available": False, "reason": "5件未満のため表示しません", "points": []}
    vectors = np.asarray([item.embedding for item in analyses], dtype=float)
    if len(comments) >= 30:
        try:
            import umap

            reducer = umap.UMAP(
                n_neighbors=min(30, max(5, int(math.sqrt(len(comments))))),
                min_dist=0.08,
                metric="cosine",
                random_state=settings.umap_random_state,
            )
            coordinates = reducer.fit_transform(vectors)
            reducer_name = "UMAP"
        except ImportError:
            with threadpool_limits(limits=1):
                coordinates = PCA(n_components=2, random_state=42).fit_transform(
                    vectors
                )
            reducer_name = "PCA fallback"
    else:
        with threadpool_limits(limits=1):
            coordinates = PCA(n_components=2, random_state=42).fit_transform(vectors)
        reducer_name = "PCA"
    labels, clusterer_name = _cluster(vectors, settings)
    counts = Counter(int(label) for label in labels)
    centers = {
        label: np.mean(vectors[labels == label], axis=0)
        for label in sorted(set(int(value) for value in labels))
    }
    points = []
    for index, (comment, analysis) in enumerate(zip(comments, analyses, strict=True)):
        label = int(labels[index])
        distances = np.linalg.norm(vectors - vectors[index], axis=1)
        neighbors = np.argsort(distances)[1:6]
        local_density = float(np.mean(distances[neighbors])) if len(neighbors) else 0
        originality = min(1.0, local_density / max(0.001, float(np.mean(distances))))
        center_distance = float(np.linalg.norm(vectors[index] - centers[label]))
        analysis.cluster_id = f"cluster-{label}" if label >= 0 else "noise"
        analysis.originality_score = originality
        points.append(
            {
                "comment_id": comment.comment_id,
                "x": float(coordinates[index, 0]),
                "y": float(coordinates[index, 1]),
                "cluster_id": analysis.cluster_id,
                "cluster_label": f"クラスター {label + 1}" if label >= 0 else "ノイズ",
                "cluster_share": counts[label] / len(comments),
                "empathy_count": comment.empathy_count or 0,
                "dominant_emotion": analysis.dominant_emotion,
                "originality": originality,
                "relevance": analysis.relevance_score,
                "center_distance": center_distance,
                "is_minority": counts[label] / len(comments)
                <= settings.minority_cluster_share_max,
            }
        )
    return {
        "available": True,
        "reducer": reducer_name,
        "clusterer": clusterer_name,
        "points": points,
    }
