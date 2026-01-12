from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from .confluence_client import ConfluenceClient, ConfluenceClientConfig

log = logging.getLogger(__name__)


@dataclass
class PageRecord:
    page_id: str
    title: str
    space_key: str
    status: Optional[str]
    version: Optional[int]
    last_updated: Optional[str]
    url_full: Optional[str]
    url_short: Optional[str]
    ancestors: List[Dict[str, Any]]
    body_storage: str
    body_view: str
    body_export_view: str
    labels: List[str]
    hash: str

    def to_page_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("hash", None)
        return data

    def to_index_json(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "title": self.title,
            "space_key": self.space_key,
            "url_full": self.url_full,
            "url_short": self.url_short,
            "last_updated": self.last_updated,
            "hash": self.hash,
        }


def _build_urls(base_url: str, links: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    webui = links.get("webui")
    tinyui = links.get("tinyui")
    full_url = urljoin(base_url, webui) if webui else None
    short_url = urljoin(base_url, tinyui) if tinyui else None
    return full_url, short_url


def _extract_page_record(raw: Dict[str, Any], base_url: str) -> PageRecord:
    page_id = str(raw.get("id"))
    body = raw.get("body", {}) or {}
    body_storage = body.get("storage", {}).get("value", "") or ""
    body_view = body.get("view", {}).get("value", "") or ""
    body_export_view = body.get("export_view", {}).get("value", "") or ""
    version_info = raw.get("version") or {}
    links = raw.get("_links") or {}
    url_full, url_short = _build_urls(base_url, links)
    ancestors_raw = raw.get("ancestors") or []
    ancestors = [{"id": str(a.get("id")), "title": a.get("title")} for a in ancestors_raw]
    labels_raw = raw.get("metadata", {}).get("labels", {}).get("results", []) or []
    labels = [lbl.get("name") for lbl in labels_raw if lbl.get("name")]
    content_hash = sha256(body_storage.encode("utf-8")).hexdigest()
    return PageRecord(
        page_id=page_id,
        title=raw.get("title") or "",
        space_key=raw.get("space", {}).get("key") or "",
        status=raw.get("status"),
        version=version_info.get("number"),
        last_updated=version_info.get("when"),
        url_full=url_full,
        url_short=url_short,
        ancestors=ancestors,
        body_storage=body_storage,
        body_view=body_view,
        body_export_view=body_export_view,
        labels=labels,
        hash=content_hash,
    )


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def scrape_space(space_key: str, root_page_id: str, out_dir: Path) -> None:
    config = ConfluenceClientConfig.from_env()
    client = ConfluenceClient(config)
    base_space_dir = Path(out_dir) / space_key
    index_file = base_space_dir / "index.jsonl"

    log.info("Scraping Confluence space %s starting at page %s", space_key, root_page_id)
    pages: List[PageRecord] = []

    # include root page
    root_raw = client.get_page(root_page_id)
    pages.append(_extract_page_record(root_raw, config.url))

    for raw in client.iter_descendants(root_page_id):
        space = (raw.get("space") or {}).get("key")
        if space and space != space_key:
            continue
        pages.append(_extract_page_record(raw, config.url))

    # clear index file if exists
    if index_file.exists():
        index_file.unlink()

    for page in pages:
        page_path = base_space_dir / f"{page.page_id}.json"
        _write_json(page_path, page.to_page_json())
        _append_jsonl(index_file, page.to_index_json())
        log.info("Saved page %s (%s)", page.page_id, page.title)

    log.info("Completed scraping %s pages into %s", len(pages), base_space_dir)
