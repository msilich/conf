from __future__ import annotations

from typing import Iterable, List

from .base import EmbeddingsProvider


class WatsonxEmbeddingsProvider(EmbeddingsProvider):
    """
    Skeleton provider for IBM watsonx embeddings.
    Configure credentials and endpoint before use.
    """

    def __init__(self, dimension: int, model: str | None = None):
        super().__init__(dimension)
        self.model = model or "ibm/granite-embed"

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        raise NotImplementedError(
            "WatsonxEmbeddingsProvider is a skeleton. Add API integration here."
        )
