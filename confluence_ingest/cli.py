import argparse
import logging
import os
import sys
from pathlib import Path

from .scraper import scrape_space
from .loader import load_directory


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="confluence_ingest",
        description="Scrape Confluence pages and load them into PostgreSQL with pgvector.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    scrape = sub.add_parser("scrape", help="Scrape Confluence pages into JSON files")
    scrape.add_argument("--space-key", required=True, help="Confluence space key, e.g. IT4PLT")
    scrape.add_argument(
        "--root-page-id",
        type=str,
        help="Root page id to start scraping (defaults to CONF_ROOT_PAGE_ID env var)",
    )
    scrape.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for scraped JSON",
    )

    load = sub.add_parser("load", help="Load scraped JSON into PostgreSQL with embeddings")
    load.add_argument("--in-dir", type=Path, required=True, help="Input directory for JSON files")
    load.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    load.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    load.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform all steps except writing to the database",
    )

    full = sub.add_parser("full", help="Run scrape then load with matching directories")
    full.add_argument("--space-key", required=True, help="Confluence space key, e.g. IT4PLT")
    full.add_argument(
        "--root-page-id",
        type=str,
        help="Root page id to start scraping (defaults to CONF_ROOT_PAGE_ID env var)",
    )
    full.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Base data directory",
    )
    full.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    full.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    full.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform all steps except writing to the database",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "scrape":
        root_page = args.root_page_id or os.getenv("CONF_ROOT_PAGE_ID")
        if not root_page:
            parser.error("Root page id must be provided via --root-page-id or CONF_ROOT_PAGE_ID")
        scrape_space(
            space_key=args.space_key,
            root_page_id=str(root_page),
            out_dir=args.out_dir,
        )
    elif args.command == "load":
        load_directory(
            input_dir=args.in_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            dry_run=args.dry_run,
        )
    elif args.command == "full":
        root_page = args.root_page_id or os.getenv("CONF_ROOT_PAGE_ID")
        if not root_page:
            parser.error("Root page id must be provided via --root-page-id or CONF_ROOT_PAGE_ID")
        target_dir = args.data_dir / args.space_key
        scrape_space(
            space_key=args.space_key,
            root_page_id=str(root_page),
            out_dir=args.data_dir,
        )
        load_directory(
            input_dir=target_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            dry_run=args.dry_run,
        )
    else:
        parser.error("Unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
