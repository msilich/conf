from __future__ import annotations

import re
from bs4 import BeautifulSoup, NavigableString, Tag


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    # Remove scripts/styles that pollute text
    for tag in soup(["script", "style"]):
        tag.decompose()
    _preserve_confluence_macros(soup)
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    _preserve_confluence_macros(soup)
    root = soup.body or soup
    markdown = _render_markdown(root)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


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


def _render_markdown(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""

    name = (node.name or "").lower()
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(name[1])
        title = _render_children_markdown(node).strip()
        return f"{'#' * level} {title}\n\n"
    if name == "p":
        return f"{_render_children_markdown(node).strip()}\n\n"
    if name == "br":
        return "\n"
    if name == "pre":
        code_text = node.get_text()
        return f"```\n{code_text.rstrip()}\n```\n\n"
    if name == "code":
        return f"`{_render_children_markdown(node).strip()}`"
    if name == "a":
        label = _render_children_markdown(node).strip()
        href = node.get("href")
        return f"[{label}]({href})" if href else label
    if name in {"strong", "b"}:
        return f"**{_render_children_markdown(node).strip()}**"
    if name in {"em", "i"}:
        return f"*{_render_children_markdown(node).strip()}*"
    if name == "blockquote":
        text = _render_children_markdown(node).strip()
        text = text.replace("\n", "\n> ")
        return f"> {text}\n\n"
    if name == "ul":
        items = []
        for li in node.find_all("li", recursive=False):
            items.append(f"- {_render_children_markdown(li).strip()}")
        return "\n".join(items) + "\n\n" if items else ""
    if name == "ol":
        items = []
        for idx, li in enumerate(node.find_all("li", recursive=False), start=1):
            items.append(f"{idx}. {_render_children_markdown(li).strip()}")
        return "\n".join(items) + "\n\n" if items else ""
    if name == "table":
        rows = []
        for row in node.find_all("tr", recursive=False):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"], recursive=False)]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows) + "\n\n" if rows else ""

    return _render_children_markdown(node)


def _render_children_markdown(node: Tag) -> str:
    parts = []
    for child in node.children:
        parts.append(_render_markdown(child))
    return "".join(parts)
