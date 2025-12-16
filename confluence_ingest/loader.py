from __future__ import annotations

import json
import logging
import os
from hashlib import sha256
from pathlib import Path
from typing import Dict, List

from .chunker import chunk_html
from .db import delete_page_chunks, ensure_schema, ensure_vector_index, fetch_page_hashes, get_connection, insert_chunks
from .embeddings.base import EmbeddingsProvider, embeddings_dimension_from_env
from .embeddings.dummy import DummyEmbeddingsProvider
from .embeddings.ollama_provider import OllamaEmbeddingsProvider
from .embeddings.openai_provider import OpenAIEmbeddingsProvider
from .embeddings.watsonx_provider import WatsonxEmbeddingsProvider

log = logging.getLogger(__name__)


def _provider_from_env(dimension: int) -> EmbeddingsProvider:
    provider_name = os.getenv("EMBEDDINGS_PROVIDER", "dummy").lower()
    if provider_name == "dummy":
        return DummyEmbeddingsProvider(dimension)
    if provider_name == "openai":
        return OpenAIEmbeddingsProvider(dimension)
    if provider_name == "ollama":
        return OllamaEmbeddingsProvider(dimension)
    if provider_name == "watsonx":
        return WatsonxEmbeddingsProvider(dimension)
    raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER '{provider_name}'")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _hash_content(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _embeddings_or_raise(provider: EmbeddingsProvider, chunks: List[str]) -> List[List[float]]:
    vectors = provider.embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise ValueError("Embeddings provider returned mismatched length")
    for vec in vectors:
        if len(vec) != provider.dimension:
            raise ValueError("Embeddings dimension mismatch from provider")
    return vectors


def _page_changed(existing_hashes: Dict[int, str], new_hashes: Dict[int, str]) -> bool:
    if not existing_hashes:
        return True
    if len(existing_hashes) != len(new_hashes):
        return True
    for cid, h in new_hashes.items():
        if existing_hashes.get(cid) != h:
            return True
    return False


def load_directory(input_dir: Path, chunk_size: int, chunk_overlap: int, dry_run: bool = False) -> None:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    dimension = embeddings_dimension_from_env()
    provider = _provider_from_env(dimension)
    log.info("Using embeddings provider '%s' with dim=%s", provider.__class__.__name__, dimension)

    json_files = sorted(p for p in input_dir.glob("*.json") if p.name != "index.jsonl")
    if not json_files:
        log.warning("No JSON files found in %s", input_dir)
        return

    conn = None
    if not dry_run:
        conn = get_connection()
        ensure_schema(conn, dimension)
        ensure_vector_index(conn)

    for page_file in json_files:
        page = _load_json(page_file)
        page_id = str(page.get("page_id"))
        title = page.get("title")
        url_full = page.get("url_full")
        url_short = page.get("url_short")
        updated_at = page.get("last_updated")
        html = page.get("body_storage") or ""

        chunks = chunk_html(html, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunk_hashes = {idx: _hash_content(chunk) for idx, chunk in enumerate(chunks)}

        if dry_run:
            log.info("[dry-run] Would load page %s with %s chunks", page_id, len(chunks))
            continue

        assert conn is not None
        existing_hashes = fetch_page_hashes(conn, page_id)
        if not _page_changed(existing_hashes, chunk_hashes):
            log.info("Skipping unchanged page %s", page_id)
            continue

        log.info("Loading page %s (%s chunks)", page_id, len(chunks))
        delete_page_chunks(conn, page_id)

        vectors = _embeddings_or_raise(provider, chunks)
        rows: List[dict] = []
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            rows.append(
                {
                    "space_key": page.get("space_key"),
                    "page_id": page_id,
                    "chunk_id": idx,
                    "title": title,
                    "url_full": url_full,
                    "url_short": url_short,
                    "updated_at": updated_at,
                    "content": chunk,
                    "content_hash": chunk_hashes[idx],
                    "embedding": vec,
                    "metadata": {
                        "ancestors": page.get("ancestors"),
                        "labels": page.get("labels"),
                        "status": page.get("status"),
                        "version": page.get("version"),
                    },
                }
            )

        insert_chunks(conn, rows)

    if conn:
        conn.close()
