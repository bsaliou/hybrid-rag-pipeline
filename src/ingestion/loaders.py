"""
Chargement de documents réels (PDF, HTML, Markdown) vers une représentation
texte unifiée, avec métadonnées de provenance (fichier, page, titre de section).

Design : chaque loader retourne une liste de `RawDocument`, une unité par
"section logique" quand c'est possible (page PDF, section HTML/MD), plutôt
qu'un blob unique. Ça donne un meilleur point de départ au chunker sémantique
et ça permet de tracer précisément la source dans les citations finales.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class RawDocument:
    text: str
    source: str                      # nom du fichier
    metadata: dict[str, Any] = field(default_factory=dict)


def load_pdf(path: str | Path) -> list[RawDocument]:
    """Une RawDocument par page, avec le numéro de page en métadonnée."""
    path = Path(path)
    reader = PdfReader(str(path))
    docs: list[RawDocument] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = _clean_text(text)
        if not text.strip():
            continue
        docs.append(
            RawDocument(
                text=text,
                source=path.name,
                metadata={"page": i + 1, "doc_type": "pdf"},
            )
        )
    return docs


def load_html(path: str | Path) -> list[RawDocument]:
    """
    Découpe le HTML par sections (h1/h2/h3) plutôt que de tout aplatir en un
    seul bloc : ça préserve la structure logique du document pour le chunking.
    """
    path = Path(path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")

    # on retire le bruit non-informatif
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    docs: list[RawDocument] = []
    current_title = "Introduction"
    current_parts: list[str] = []

    def flush():
        text = _clean_text(" ".join(current_parts))
        if text.strip():
            docs.append(
                RawDocument(
                    text=text,
                    source=path.name,
                    metadata={"section_title": current_title, "doc_type": "html"},
                )
            )

    body = soup.body or soup
    for el in body.descendants:
        if getattr(el, "name", None) in ("h1", "h2", "h3"):
            flush()
            current_title = el.get_text(strip=True) or current_title
            current_parts = []
        elif getattr(el, "name", None) in ("p", "li", "td"):
            txt = el.get_text(" ", strip=True)
            if txt:
                current_parts.append(txt)
    flush()

    if not docs:  # fallback si aucune structure de titres détectée
        text = _clean_text(soup.get_text(" ", strip=True))
        docs.append(RawDocument(text=text, source=path.name, metadata={"doc_type": "html"}))

    return docs


def load_markdown(path: str | Path) -> list[RawDocument]:
    """Découpe le Markdown par en-têtes (#, ##, ###)."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")

    header_pattern = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
    matches = list(header_pattern.finditer(raw))

    docs: list[RawDocument] = []
    if not matches:
        text = _clean_text(raw)
        return [RawDocument(text=text, source=path.name, metadata={"doc_type": "markdown"})]

    for idx, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        body = raw[start:end]
        text = _clean_text(_strip_markdown_syntax(body))
        if text.strip():
            docs.append(
                RawDocument(
                    text=text,
                    source=path.name,
                    metadata={"section_title": title, "doc_type": "markdown"},
                )
            )
    return docs


def load_document(path: str | Path) -> list[RawDocument]:
    """Point d'entrée unique : dispatch selon l'extension du fichier."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".html", ".htm"):
        return load_html(path)
    if suffix in (".md", ".markdown"):
        return load_markdown(path)
    if suffix == ".txt":
        text = _clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        return [RawDocument(text=text, source=path.name, metadata={"doc_type": "txt"})]
    raise ValueError(f"Format non supporté : {suffix}")


def load_directory(dir_path: str | Path) -> list[RawDocument]:
    """Charge récursivement tous les fichiers supportés d'un dossier."""
    dir_path = Path(dir_path)
    supported = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt"}
    docs: list[RawDocument] = []
    for p in sorted(dir_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in supported:
            docs.extend(load_document(p))
    return docs


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # recolle les mots coupés par un tiret PDF
    return text.strip()


def _strip_markdown_syntax(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # blocs de code
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # liens -> texte du lien
    text = re.sub(r"[*_>#]", "", text)
    return text
