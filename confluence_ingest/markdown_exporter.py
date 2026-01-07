from __future__ import annotations

import json
import logging
from pathlib import Path

from .text_clean import html_to_markdown

log = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def export_markdown_directory(input_dir: Path, output_dir: Path) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    json_files = sorted(p for p in input_dir.glob("*.json") if p.name != "index.jsonl")
    if not json_files:
        log.warning("No JSON files found in %s", input_dir)
        return

    for page_file in json_files:
        page = _load_json(page_file)
        page_id = str(page.get("page_id"))
        title = page.get("title") or ""
        url_full = page.get("url_full") or page.get("url_short") or ""
        html = page.get("body_storage") or ""

        body_md = html_to_markdown(html)
        header_lines = []
        if title:
            header_lines.append(f"# {title}")
        if url_full:
            header_lines.append(f"Source: {url_full}")
        header = "\n".join(header_lines).strip()
        markdown = f"{header}\n\n{body_md}\n" if header else f"{body_md}\n"

        out_path = output_dir / f"{page_id}.md"
        _write_markdown(out_path, markdown)
        log.info("Wrote %s", out_path)
