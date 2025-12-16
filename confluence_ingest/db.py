from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import psycopg


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )
    return conn


def ensure_schema(conn: psycopg.Connection, dimension: int) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS confluence_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                space_key TEXT NOT NULL,
                page_id TEXT NOT NULL,
                chunk_id INT NOT NULL,
                title TEXT,
                url_full TEXT,
                url_short TEXT,
                updated_at TIMESTAMPTZ,
                content TEXT,
                content_hash TEXT,
                embedding VECTOR({dimension}),
                metadata JSONB DEFAULT '{{}}'::jsonb,
                UNIQUE(space_key, page_id, chunk_id)
            );
            """
        )
    conn.commit()


def ensure_vector_index(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS confluence_chunks_embedding_hnsw
                ON confluence_chunks USING hnsw (embedding vector_cosine_ops);
                """
            )
        except psycopg.Error:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS confluence_chunks_embedding_ivfflat
                ON confluence_chunks USING ivfflat (embedding vector_cosine_ops);
                """
            )
    conn.commit()


def fetch_page_hashes(conn: psycopg.Connection, page_id: str) -> Dict[int, str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id, content_hash FROM confluence_chunks WHERE page_id = %s",
            (page_id,),
        )
        rows = cur.fetchall()
    return {int(row[0]): row[1] for row in rows}


def delete_page_chunks(conn: psycopg.Connection, page_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM confluence_chunks WHERE page_id = %s", (page_id,))
    conn.commit()


def insert_chunks(conn: psycopg.Connection, rows: Iterable[dict]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO confluence_chunks (
                space_key, page_id, chunk_id, title, url_full, url_short,
                updated_at, content, content_hash, embedding, metadata
            ) VALUES (
                %(space_key)s, %(page_id)s, %(chunk_id)s, %(title)s, %(url_full)s, %(url_short)s,
                %(updated_at)s, %(content)s, %(content_hash)s, %(embedding)s, %(metadata)s
            )
            ON CONFLICT (space_key, page_id, chunk_id)
            DO UPDATE SET
                title = EXCLUDED.title,
                url_full = EXCLUDED.url_full,
                url_short = EXCLUDED.url_short,
                updated_at = EXCLUDED.updated_at,
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata;
            """,
            rows,
        )
    conn.commit()
