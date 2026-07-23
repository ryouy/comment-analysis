from __future__ import annotations

import hashlib
import math

import numpy as np

from src.preprocessing.text import tokenize


def local_embedding(text: str, dimensions: int = 96) -> list[float]:
    vector = np.zeros(dimensions, dtype=float)
    tokens = tokenize(text)
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1 if digest[4] % 2 else -1
        vector[index] += sign * (1 + math.log1p(len(token)))
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector.tolist()


def cosine(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not denominator:
        return 0.0
    return max(0.0, min(1.0, (float(np.dot(a, b)) / denominator + 1) / 2))


def lexical_jaccard(left: str, right: str) -> float:
    a, b = set(tokenize(left)), set(tokenize(right))
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

