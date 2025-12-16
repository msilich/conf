from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterable, List


class EmbeddingsProvider(ABC):
    def __init__(self, dimension: int):
        self.dimension = dimension

    @abstractmethod
    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        ...


def embeddings_dimension_from_env() -> int:
    dim_raw = os.getenv("EMBEDDINGS_DIM", "768")
    try:
        dim = int(dim_raw)
    except ValueError as exc:
        raise ValueError("EMBEDDINGS_DIM must be an integer") from exc
    if dim <= 0:
        raise ValueError("EMBEDDINGS_DIM must be positive")
    return dim
