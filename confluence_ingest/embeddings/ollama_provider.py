from __future__ import annotations

from typing import Iterable, List

from .base import EmbeddingsProvider


class OllamaEmbeddingsProvider(EmbeddingsProvider):
    """
    Skeleton provider for Ollama local embeddings.
    Configure endpoint/model before use.
    """

    def __init__(self, dimension: int, model: str | None = None):
        super().__init__(dimension)
        self.model = model or "nomic-embed-text"

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        raise NotImplementedError(
            "OllamaEmbeddingsProvider is a skeleton. Integrate Ollama client call here."
        )
