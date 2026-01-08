#!/usr/bin/env python3
"""
Synthetische Q&A aus Confluence-Markdown erzeugen.

NEU:
- --suggest-prompt PATH
  PATH kann eine Datei ODER ein Ordner sein.
  - Datei: wie bisher (interaktiv Vorschläge, dann anwenden)
  - Ordner: jede .md Datei wird nacheinander interaktiv behandelt:
      1) Prompt-Vorschlag generieren
      2) Script bewertet den Vorschlag (Heuristik)
      3) User entscheidet: y anwenden / n neuer Vorschlag / q Datei überspringen / x Abbruch

Zusätzlich:
- Robustere JSON-Extraktion für Prompt-Vorschläge (kein Crash bei kaputtem JSON).
- Prompt-Suggest berücksichtigt Tabellen UND Codeblöcke; Q&A sollen Codeblöcke vollständig als Markdown übernehmen.

Beispiele:
  # Ordner interaktiv abarbeiten
  python3 synth_qa.py --suggest-prompt data/converted

  # Ordner interaktiv, feste QA-Anzahl beim Anwenden
  python3 synth_qa.py --suggest-prompt data/converted --qa 18
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx


# -------------------------
# Default-Pfade (Batch)
# -------------------------
INPUT_DIR = Path("data/converted")
OUTPUT_DIR = Path("data/synth")


# -------------------------
# QA-Heuristik
# -------------------------
MIN_QA = 4
MAX_QA = 30
WORDS_PER_QA = 180  # ca. 1 QA pro 180 Wörter

MAX_WORDS_PER_CHUNK = 1100

MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 1.5

MAX_SUGGEST_ATTEMPTS = 5


# -------------------------
# LLM Config
# -------------------------
@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout_seconds: float = 1800.0


def _httpx_verify_setting() -> object:
    ca_bundle = os.getenv("LLM_CA_BUNDLE", "").strip()
    insecure = os.getenv("LLM_INSECURE", "").strip().lower() in {"1", "true", "yes", "on"}

    if insecure:
        return False
    if ca_bundle:
        return ca_bundle
    return True


def load_llm_config() -> LLMConfig:
    base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "mistralai/mistral-small-3-1-24b-instruct-2503")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    return LLMConfig(base_url=base_url, api_key=api_key, model=model, temperature=temperature)


def _headers(cfg: LLMConfig) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if cfg.api_key:
        h["Authorization"] = f"Bearer {cfg.api_key}"
    return h


def call_chat_completions_json(cfg: LLMConfig, system_prompt: str, user_prompt: str) -> str:
    url = f"{cfg.base_url}/chat/completions"
    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    verify = _httpx_verify_setting()

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=cfg.timeout_seconds, verify=verify) as client:
                r = client.post(url, headers=_headers(cfg), json=payload)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            sleep_s = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.random()
            time.sleep(min(sleep_s, 20.0))
    raise RuntimeError(f"LLM request failed after retries: {last_err}")


# -------------------------
# Markdown parsing
# -------------------------
FRONTMATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n", re.DOTALL)
FENCED_CODEBLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)


def normalize_newlines(md: str) -> str:
    return md.replace("\r\n", "\n").replace("\r", "\n")


def split_frontmatter(md: str) -> Tuple[Optional[str], str]:
    md = normalize_newlines(md)
    m = FRONTMATTER_RE.match(md)
    if not m:
        return None, md
    frontmatter = md[: m.end()]
    body = md[m.end() :]
    return frontmatter, body


def normalize_whitespace(s: str) -> str:
    s = normalize_newlines(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip() + "\n"


def count_words(s: str) -> int:
    return len(re.findall(r"\S+", s))


def estimate_qa_pairs(word_count: int) -> int:
    if word_count <= 0:
        return MIN_QA
    est = int(round(word_count / WORDS_PER_QA))
    return max(MIN_QA, min(MAX_QA, est))


def chunk_by_words(text: str, max_words: int) -> List[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _split_markdown_blocks(md: str) -> List[str]:
    lines = normalize_newlines(md).split("\n")
    blocks: List[str] = []
    buf: List[str] = []
    in_code = False

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = line.strip().startswith("```")
        if fence:
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            in_code = not in_code
            buf.append(line)
            i += 1
            continue

        if in_code:
            buf.append(line)
            i += 1
            continue

        if (
            i + 1 < len(lines)
            and re.match(r"^\s*\|.+\|\s*$", line)
            and re.match(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$", lines[i + 1])
        ):
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and re.match(r"^\s*\|.*\|\s*$", lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append("\n".join(table_lines).strip())
            continue

        if line.strip() == "":
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            i += 1
            continue

        buf.append(line)
        i += 1

    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]


def _chunk_markdown_preserve_blocks(md: str, max_words: int) -> List[str]:
    blocks = _split_markdown_blocks(md)
    if not blocks:
        return []

    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0

    for block in blocks:
        block_words = count_words(block)
        if block_words == 0:
            continue

        if cur and cur_words + block_words > max_words:
            chunks.append("\n\n".join(cur))
            cur = []
            cur_words = 0

        if block_words > max_words:
            if cur:
                chunks.append("\n\n".join(cur))
                cur = []
                cur_words = 0
            if block.lstrip().startswith("```") or block.lstrip().startswith("|"):
                chunks.append(block)
            else:
                for part in chunk_by_words(block, max_words):
                    chunks.append(part)
            continue

        cur.append(block)
        cur_words += block_words

    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def has_markdown_table(body: str) -> bool:
    # sehr einfache Heuristik: Tabellenzeile + Separator
    return bool(re.search(r"^\s*\|.+\|\s*$\n^\s*\|(?:\s*:?-+:?\s*\|)+\s*$", body, re.MULTILINE))


def count_codeblocks(body: str) -> int:
    return len(FENCED_CODEBLOCK_RE.findall(body))


# -------------------------
# Robust JSON parsing (für Suggest)
# -------------------------
def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _safe_json_loads(text: str) -> Optional[dict]:
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    extracted = _extract_json_object(text)
    if extracted:
        try:
            return json.loads(extracted)
        except Exception:
            repaired = extracted.replace("\r\n", "\n").replace("\r", "\n")
            repaired = repaired.replace("\n", "\\n")
            try:
                return json.loads(repaired)
            except Exception:
                pass

    return None


# -------------------------
# Prompt-Building für Q&A
# -------------------------
BASE_SYSTEM_PROMPT = """Du erzeugst synthetische Frage-Antwort-Paare aus bereitgestelltem Markdown-Text.

Regeln:
- Nutze ausschließlich Informationen aus dem gegebenen Text. Nichts erfinden.
- Antworte kurz, präzise und fachlich korrekt (typisch 1–8 Sätze, plus ggf. Codeblock).
- Stelle praxisnahe Fragen (Definition, Voraussetzungen, Vorgehen, Fehlerbehebung, Grenzen).
- Berücksichtige alle relevanten Markdown-Elemente:
  - Fließtext
  - Überschriften
  - Tabellen
  - Codeblöcke (``` … ```)

Codeblöcke:
- Wenn Codeblöcke vorhanden sind (z. B. YAML, JSON, Bash, Python):
  - Erkenne deren Zweck (CLI-Befehl, Konfiguration, Beispiel, Script).
  - Erzeuge mindestens eine Frage, die sich direkt auf den Code bezieht, falls sinnvoll.
  - Übernimm den Code in der Antwort als Beispiel.
  - Wenn möglich, übernimm den Code vollständig (nicht nur Ausschnitte).
  - Verändere den Code nicht inhaltlich.
  - Formatiere Code in der Antwort als korrektes Markdown mit fenced code blocks und Sprache (```yaml, ```python, ...).

Ausgabeformat:
- Liefere strikt JSON:
  {
    "qa": [
      {"q": "...", "a": "..."}
    ]
  }

Wichtig für JSON:
- Der Wert von "a" darf Markdown enthalten (auch ```-Codeblöcke).
- Zeilenumbrüche als \\n kodieren und Anführungszeichen escapen.
"""

USER_PROMPT_TEMPLATE = """Erzeuge genau {n} Frage-Antwort-Paare aus dem folgenden Text.

Text:
\"\"\"{chunk}\"\"\"

Gib ausschließlich JSON zurück im Format:
{{
  "qa": [
    {{"q": "...", "a": "..."}}
  ]
}}
"""


def build_system_prompt(extra_rules: Optional[str]) -> str:
    if not extra_rules:
        return BASE_SYSTEM_PROMPT
    extra_rules = extra_rules.strip()
    lines = [line.strip("- ").strip() for line in extra_rules.splitlines() if line.strip()]
    return BASE_SYSTEM_PROMPT + "\n\nZusätzliche Vorgaben:\n- " + "\n- ".join(lines)


# -------------------------
# Q&A generation
# -------------------------
def dedupe_qa(qa: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for item in qa:
        q = (item.get("q") or "").strip()
        a = (item.get("a") or "").strip()
        if not q or not a:
            continue
        key = re.sub(r"\s+", " ", q.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"q": q, "a": a})
    return out


def distribute_pairs_across_chunks(chunks: List[str], target_pairs: int) -> List[int]:
    counts = [count_words(c) for c in chunks]
    total = sum(counts) or 1
    per = [max(1, int(round(target_pairs * (c / total)))) for c in counts]

    cur = sum(per)
    while cur > target_pairs:
        idx = max(range(len(per)), key=lambda i: per[i])
        if per[idx] > 1:
            per[idx] -= 1
            cur -= 1
        else:
            break
    while cur < target_pairs:
        idx = min(range(len(per)), key=lambda i: per[i])
        per[idx] += 1
        cur += 1
    return per


def generate_qa(cfg: LLMConfig, md_body: str, target_pairs: int, extra_rules: Optional[str] = None) -> List[Dict[str, str]]:
    md_body = normalize_whitespace(md_body)
    chunks = _chunk_markdown_preserve_blocks(md_body, MAX_WORDS_PER_CHUNK)
    if not chunks:
        return []

    per_chunk = distribute_pairs_across_chunks(chunks, target_pairs)
    system_prompt = build_system_prompt(extra_rules)

    all_qa: List[Dict[str, str]] = []
    for chunk, n in zip(chunks, per_chunk):
        content = call_chat_completions_json(
            cfg,
            system_prompt=system_prompt,
            user_prompt=USER_PROMPT_TEMPLATE.format(n=n, chunk=chunk),
        )
        obj = _safe_json_loads(content)
        if not obj:
            continue
        qa_list = obj.get("qa", [])
        if not isinstance(qa_list, list):
            continue
        for it in qa_list:
            if isinstance(it, dict):
                all_qa.append({"q": str(it.get("q", "")).strip(), "a": str(it.get("a", "")).strip()})

    all_qa = dedupe_qa(all_qa)
    return all_qa[: max(MIN_QA, min(MAX_QA, target_pairs))]


# -------------------------
# Prompt suggestion (full-file analysis)
# -------------------------
PROMPT_SUGGEST_SYSTEM = """Du bist ein Experte für Wissensaufbereitung und RAG.

Aufgabe:
- Analysiere den bereitgestellten Markdown-Inhalt (Fließtext + Tabellen + Codeblöcke).
- Gib einen "Zusatz-Prompt" zurück, der anschließend verwendet werden kann, um aus genau diesem Dokument besonders gute Frage-Antwort-Paare zu erzeugen.

Anforderungen an den Zusatz-Prompt:
- Formuliere ihn als kurze Liste konkreter Regeln (Bullet-Points).
- Beziehe dich auf die Struktur und Eigenheiten des Dokuments:
  - wichtige Begriffe und Abkürzungen
  - Tabellenlogik (Spalten, Bedeutung)
  - typische How-to Schritte
  - Fehlerfälle und Troubleshooting
  - Codeblöcke: welche Q&A dazu entstehen sollen; Code in Antworten vollständig als Markdown-Codeblock übernehmen
- Keine Inhalte erfinden, nur ableiten, worauf beim Extrahieren zu achten ist.

Wichtig:
- Gib NUR JSON zurück (kein zusätzlicher Text).
- Ausgabe strikt als JSON:
  {
    "extra_prompt": "..."
  }
- Der Wert von "extra_prompt" ist ein JSON-String:
  - Zeilenumbrüche als \\n
  - Anführungszeichen escapen
"""

PROMPT_SUGGEST_USER = """Analysiere dieses Dokument und liefere den besten Zusatz-Prompt, um daraus hochwertige Q&A zu erzeugen.

Dokument (Markdown):
\"\"\"{doc}\"\"\"
"""


def suggest_extra_prompt(cfg: LLMConfig, md_body_full: str, attempt: int) -> str:
    md_body_full = normalize_whitespace(md_body_full)

    user_prompt = PROMPT_SUGGEST_USER.format(doc=md_body_full)
    if attempt > 1:
        user_prompt += (
            "\n\nZusatz: Erzeuge einen ALTERNATIVEN Prompt-Vorschlag, "
            "mit anderer Gewichtung/Schwerpunktsetzung als zuvor (ohne zu wiederholen)."
        )

    content = call_chat_completions_json(cfg, PROMPT_SUGGEST_SYSTEM, user_prompt)
    obj = _safe_json_loads(content)
    if not obj:
        return (content or "").strip()
    return str(obj.get("extra_prompt", "")).strip() or (content or "").strip()


# -------------------------
# Bewertung des Prompt-Vorschlags (Heuristik)
# -------------------------
def evaluate_prompt_quality(extra_prompt: str, md_body: str) -> Tuple[str, List[str]]:
    """
    Sehr einfache Heuristik:
    - prüft Mindestlänge, Bullet-Struktur
    - prüft, ob Tabellen/Codeblöcke erwähnt werden, falls vorhanden
    - Ergebnis: ("gut"|"mittel"|"schlecht", begruendungen[])
    """
    reasons: List[str] = []
    score = 0

    text = (extra_prompt or "").strip()
    if len(text) < 80:
        reasons.append("Prompt ist sehr kurz; eventuell zu unspezifisch.")
    else:
        score += 1

    bullet_like = sum(1 for line in text.splitlines() if line.strip().startswith(("-", "*")))
    if bullet_like >= 3:
        score += 2
    else:
        reasons.append("Wenig Bullet-Points; konkrete Regeln könnten fehlen.")

    need_table = has_markdown_table(md_body)
    need_code = count_codeblocks(md_body) > 0

    lower = text.lower()

    if need_table:
        if "tabelle" in lower or "spalte" in lower or "tabellen" in lower:
            score += 2
        else:
            reasons.append("Dokument enthält Tabellen, Prompt erwähnt Tabellen/Spalten nicht.")

    if need_code:
        if "code" in lower or "codeblock" in lower or "codeblöck" in lower or "```" in text:
            score += 2
        else:
            reasons.append("Dokument enthält Codeblöcke, Prompt erwähnt Codeblöcke/Übernahme nicht.")
        if "vollständig" in lower or "komplett" in lower:
            score += 1
        else:
            reasons.append("Prompt sagt nicht explizit, dass Code vollständig übernommen werden soll.")

    if "nichts erfinden" in lower or "nur aus dem text" in lower:
        score += 1
    else:
        reasons.append("Prompt betont nicht klar genug, dass nichts erfunden werden darf.")

    if score >= 6:
        return "gut", reasons
    if score >= 4:
        return "mittel", reasons
    return "schlecht", reasons


# -------------------------
# Output rendering
# -------------------------
def render_markdown(frontmatter: Optional[str], original_body: str, qa: List[Dict[str, str]]) -> str:
    out: List[str] = []

    if frontmatter:
        fm = normalize_newlines(frontmatter).rstrip("\n") + "\n\n"
        out.append(fm)

    out.append("# Originaldokumentation\n\n")
    out.append(normalize_whitespace(original_body))
    out.append("\n---\n\n")

    out.append("# Synthetische Q&A\n\n")
    for i, item in enumerate(qa, start=1):
        out.append(f"## Frage {i}\n\n")
        out.append(f"**Q:** {item['q'].strip()}\n\n")
        out.append(f"**A:** {item['a'].strip()}\n\n")

    return normalize_whitespace("".join(out))


# -------------------------
# Verarbeitung
# -------------------------
def read_raw(in_path: Path) -> str:
    return in_path.read_text(encoding="utf-8", errors="replace")


def read_body_for_analysis(in_path: Path) -> str:
    raw = normalize_whitespace(read_raw(in_path))
    _, body = split_frontmatter(raw)
    return normalize_whitespace(body)


def process_file(cfg: LLMConfig, in_path: Path, out_dir: Path, qa_override: Optional[int], extra_rules: Optional[str]) -> None:
    raw = normalize_whitespace(read_raw(in_path))
    fm, body = split_frontmatter(raw)

    body_norm = normalize_whitespace(body)
    wc = count_words(body_norm)

    if qa_override is not None:
        target = max(1, qa_override)
    else:
        target = estimate_qa_pairs(wc)

    print(f"\n[{in_path.name}] Wörter={wc} -> Ziel QA={target}")
    qa = generate_qa(cfg, body_norm, target, extra_rules=extra_rules)

    if not qa:
        print(f"[{in_path.name}] WARN: Keine QA generiert, übersprungen.")
        return

    out_text = render_markdown(fm, body_norm, qa)
    out_path = out_dir / in_path.name
    out_path.write_text(out_text, encoding="utf-8")
    print(f"[{in_path.name}] OK: {len(qa)} QA -> {out_path}")


# -------------------------
# Interaktive Suggest-Loop pro Datei
# -------------------------
def _ask_choice(prompt: str) -> str:
    while True:
        ans = input(prompt).strip().lower()
        if ans in {"y", "n", "q", "x"}:
            return ans
        print("Bitte eingeben: y=anwenden, n=neuer Vorschlag, q=Datei überspringen, x=Abbruch.")


def interactive_suggest_and_apply_for_file(cfg: LLMConfig, md_path: Path, out_dir: Path, qa_override: Optional[int]) -> None:
    body = read_body_for_analysis(md_path)
    wc = count_words(body)
    cb = count_codeblocks(body)
    tbl = has_markdown_table(body)

    print("\n" + "=" * 80)
    print(f"Datei: {md_path}")
    print(f"Analyse: Wörter={wc}, Tabellen={'ja' if tbl else 'nein'}, Codeblöcke={cb}")
    print("=" * 80)

    chosen_extra: Optional[str] = None

    for attempt in range(1, MAX_SUGGEST_ATTEMPTS + 1):
        extra = suggest_extra_prompt(cfg, body, attempt=attempt)
        if not extra:
            print("WARN: Kein Prompt-Vorschlag erzeugt.")
            continue

        rating, reasons = evaluate_prompt_quality(extra, body)

        print(f"\n--- VORSCHLAG {attempt}/{MAX_SUGGEST_ATTEMPTS} ---")
        print(f"Bewertung: {rating}")
        if reasons:
            print("Hinweise:")
            for r in reasons[:6]:
                print(f"- {r}")
        if rating in {"schlecht", "mittel"} and attempt < MAX_SUGGEST_ATTEMPTS:
            print("Empfehlung: eher noch einen weiteren Vorschlag generieren (n).")
        print("\nZusatz-Prompt:\n")
        print(extra)
        print("\n--- ENDE ---\n")

        ans = _ask_choice("Aktion: (y=anwenden / n=neuer Vorschlag / q=überspringen / x=abbrechen): ")
        if ans == "y":
            chosen_extra = extra
            break
        if ans == "q":
            print("Übersprungen.")
            return
        if ans == "x":
            raise KeyboardInterrupt()
        # ans == "n" -> nächste Runde

    if not chosen_extra:
        print("Kein Vorschlag ausgewählt, Datei übersprungen.")
        return

    print("Anwenden des ausgewählten Zusatz-Prompts ...")
    process_file(cfg, md_path, out_dir, qa_override=qa_override, extra_rules=chosen_extra)


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Erzeuge synthetische Q&A aus Confluence-Markdown.")
    p.add_argument("--file", type=str, default=None, help="Optional: Pfad zu genau einer Markdown-Datei (statt Batch).")
    p.add_argument("--qa", type=int, default=None, help="Optional: feste Anzahl Q&A-Paare (Override).")
    p.add_argument("--extra", type=str, default=None, help="Optional: zusätzlicher Prompt-Text (Regeln/Schwerpunkte).")
    p.add_argument("--in-dir", type=str, default=str(INPUT_DIR), help="Input-Ordner für Batch (Default: data/converted).")
    p.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR), help="Output-Ordner (Default: data/synth).")

    p.add_argument(
        "--suggest-prompt",
        type=str,
        default=None,
        metavar="PATH",
        help="PATH kann Datei oder Ordner sein. Interaktiver Prompt-Vorschlag je Datei; bei 'y' anwenden.",
    )
    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main() -> int:
    args = parse_args()
    cfg = load_llm_config()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"LLM: base_url={cfg.base_url}, model={cfg.model}")

    # Suggest mode (Datei oder Ordner)
    if args.suggest_prompt:
        path = Path(args.suggest_prompt)
        if not path.exists():
            print(f"ERROR: Pfad nicht gefunden: {path}")
            return 2

        try:
            if path.is_file():
                interactive_suggest_and_apply_for_file(cfg, path, out_dir, qa_override=args.qa)
                print("\nFertig.")
                return 0

            # Ordner
            md_files = sorted(path.glob("*.md"))
            if not md_files:
                print(f"Keine .md Dateien gefunden in {path}")
                return 0

            print(f"Interaktiv: {len(md_files)} Dateien in {path}")
            for i, md in enumerate(md_files, start=1):
                print(f"\n[{i}/{len(md_files)}] Start: {md.name}")
                interactive_suggest_and_apply_for_file(cfg, md, out_dir, qa_override=args.qa)

            print("\nFertig.")
            return 0

        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            return 0

    # Single file mode
    if args.file:
        in_path = Path(args.file)
        if not in_path.exists():
            print(f"ERROR: Datei nicht gefunden: {in_path}")
            return 2
        process_file(cfg, in_path, out_dir, qa_override=args.qa, extra_rules=args.extra)
        print("\nFertig.")
        return 0

    # Batch mode (ohne Suggest)
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        print(f"ERROR: Input-Ordner nicht gefunden: {in_dir}")
        return 2

    files = sorted(in_dir.glob("*.md"))
    if not files:
        print(f"Keine .md Dateien gefunden in {in_dir}")
        return 0

    print(f"Batch: {len(files)} Dateien aus {in_dir} -> {out_dir}")
    for f in files:
        try:
            process_file(cfg, f, out_dir, qa_override=args.qa, extra_rules=args.extra)
        except Exception as e:
            print(f"[{f.name}] ERROR: {e}")

    print("\nFertig.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
