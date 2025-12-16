from __future__ import annotations

from typing import Iterable, List

from .base import EmbeddingsProvider


class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    """
    Skeleton provider for OpenAI embeddings.
    Requires environment variables such as OPENAI_API_KEY and model configuration.
    """

    def __init__(self, dimension: int, model: str | None = None):
        super().__init__(dimension)
        self.model = model or "text-embedding-3-small"

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        raise NotImplementedError(
            "OpenAIEmbeddingsProvider is a skeleton. Plug in openai client here."
        )
