# Confluence Ingest

CLI zum Scrapen von Confluence-Seiten und Laden in PostgreSQL/pgvector.

## Setup
- Python 3.11/3.12
- Optional: `python -m venv venv && source venv/bin/activate`
- Install: `pip install -e .`

## Environment
- `CONF_URL` (z. B. https://confluence.company.dir)
- `CONF_TOKEN` (Personal Access Token; wird nicht geloggt)
- `CONF_ROOT_PAGE_ID` (Fallback für CLI-Flag)
- `REQUESTS_CA_BUNDLE` (Pfad zu CA-Bundle, falls nötig)
- PostgreSQL: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`
- Embeddings: `EMBEDDINGS_PROVIDER` (`dummy`|`openai`|`ollama`|`watsonx`), `EMBEDDINGS_DIM` (Default 768)

## Nutzung
- Scrapen: `python -m confluence_ingest scrape --space-key IT4PLT --root-page-id 130840771 --out-dir data/raw`
- Laden: `python -m confluence_ingest load --in-dir data/raw/IT4PLT --chunk-size 1200 --chunk-overlap 200`
- Full-Run: `python -m confluence_ingest full --space-key IT4PLT --root-page-id 130840771 --data-dir data/raw`
- Dry-Run (kein DB-Write): `--dry-run`

## Hinweise
- JSON pro Page unter `data/raw/<space>/<page_id>.json`, Metadaten in `index.jsonl`.
- CA-Bundle via `REQUESTS_CA_BUNDLE` wird von `requests`/`atlassian-python-api` respektiert.
- pgvector wird automatisch erzeugt; Index versucht HNSW und fällt sonst auf IVFFlat zurück.
