from __future__ import annotations

import hashlib
import itertools
from typing import Iterable, List

from .base import EmbeddingsProvider


class DummyEmbeddingsProvider(EmbeddingsProvider):
    """
    Deterministic hash-based embeddings useful for testing and dry runs.
    """

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            # repeat the digest to fill dimension
            bytes_iter = itertools.cycle(h)
            vec = []
            for _ in range(self.dimension):
                vec.append(next(bytes_iter) / 255.0)
            vectors.append(vec)
        return vectors
