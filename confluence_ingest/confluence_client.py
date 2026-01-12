from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from atlassian import Confluence
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


def _build_session() -> Session:
    session = Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@dataclass
class ConfluenceClientConfig:
    url: str
    token: str

    @classmethod
    def from_env(cls) -> "ConfluenceClientConfig":
        url = os.getenv("CONF_URL")
        token = os.getenv("CONF_TOKEN")
        if not url or not token:
            raise RuntimeError("CONF_URL and CONF_TOKEN must be set in environment")
        return cls(url=url, token=token)


class ConfluenceClient:
    def __init__(self, config: ConfluenceClientConfig):
        self.config = config
        self.session = _build_session()
        self.client = Confluence(
            url=config.url,
            token=config.token,
            cloud=False,
            session=self.session,
        )

    def get_page(self, page_id: str) -> Dict[str, Any]:
        return self._with_retries(
            lambda: self.client.get_page_by_id(
                page_id,
                expand="ancestors,body.storage,body.view,body.export_view,version,space,_links,metadata.labels",
            )
        )

    def iter_descendants(self, root_page_id: str, batch_size: int = 50) -> Iterable[Dict[str, Any]]:
        queue: List[str] = [root_page_id]
        seen: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            children = self._fetch_children(current, batch_size=batch_size)
            for child in children:
                queue.append(str(child["id"]))
                yield child

    def _fetch_children(self, page_id: str, batch_size: int = 50) -> List[Dict[str, Any]]:
        start = 0
        results: List[Dict[str, Any]] = []
        while True:
            resp = self._with_retries(
                lambda: self.client.get_page_child_by_type(
                    page_id,
                    type="page",
                    start=start,
                    limit=batch_size,
                    expand="ancestors,body.storage,body.view,body.export_view,version,space,_links,metadata.labels",
                )
            )
            values = resp.get("results", []) if isinstance(resp, dict) else resp
            if not values:
                break
            results.extend(values)
            start += batch_size
            if len(values) < batch_size:
                break
        return results

    def _with_retries(self, func):
        delay = 1.0
        for attempt in range(6):
            try:
                return func()
            except Exception as exc:  # broad to bubble up retries
                if attempt >= 5:
                    raise
                log.warning("Confluence API call failed (attempt %s): %s", attempt + 1, exc)
                time.sleep(delay)
                delay *= 2
