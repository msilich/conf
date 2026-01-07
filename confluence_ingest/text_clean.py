from __future__ import annotations

import re
from bs4 import BeautifulSoup


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    # Remove scripts/styles that pollute text
    for tag in soup(["script", "style"]):
        tag.decompose()
    _preserve_confluence_macros(soup)
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _preserve_confluence_macros(soup: BeautifulSoup) -> None:
    macro_tags = soup.find_all(lambda t: t.name in {"ac:structured-macro", "ac:macro"})
    for macro in macro_tags:
        macro_name = macro.get("ac:name") or macro.get("name") or "macro"
        plain_body = macro.find(lambda t: t.name in {"ac:plain-text-body", "plain-text-body"})
        rich_body = macro.find(lambda t: t.name in {"ac:rich-text-body", "rich-text-body"})

        if macro_name in {"code", "codeblock", "code-block"}:
            code_text = ""
            if plain_body:
                code_text = plain_body.get_text()
            else:
                code_text = macro.get_text()
            pre = soup.new_tag("pre")
            pre.string = code_text
            macro.replace_with(pre)
            continue

        if rich_body:
            rich_body.unwrap()
            macro.unwrap()
            continue

        if plain_body:
            macro.replace_with(plain_body.get_text())
            continue

        macro.replace_with(f"[macro:{macro_name}]")
