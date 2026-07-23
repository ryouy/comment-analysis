from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.analysis.vector import local_embedding
from src.config.settings import Settings


class EmbeddingClient(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class LocalEmbeddingClient:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [local_embedding(text) for text in texts]


class OpenAIEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_request_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.settings.embedding_batch_size):
            batch = list(texts[start : start + self.settings.embedding_batch_size])
            if not batch:
                continue
            response = self.client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=batch,
                encoding_format="float",
            )
            vectors.extend(item.embedding for item in response.data)
        if len(vectors) != len(texts):
            raise ValueError("OpenAI embedding response count mismatch")
        return vectors
