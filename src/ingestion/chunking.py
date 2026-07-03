"""
Chunking sémantique.

Pourquoi pas un simple split par N tokens ?
--------------------------------------------
Un split fixe coupe des idées en plein milieu (une phrase clé peut se
retrouver scindée entre deux chunks), ce qui dégrade le rappel au retrieval
et la fidélité de la génération. L'approche ici :

1. On segmente le texte en phrases.
2. On calcule l'embedding de chaque phrase (modèle léger, local).
3. On regarde la similarité cosinus entre phrases consécutives : une chute
   de similarité marque une frontière sémantique probable (changement de
   sujet). On regroupe les phrases entre deux frontières.
4. On applique ensuite des garde-fous de taille (min/max tokens) pour éviter
   les micro-chunks inexploitables ou les chunks trop gros pour la fenêtre
   de contexte du reranker/LLM — un groupe trop long est re-scindé, un
   groupe trop court est fusionné avec son voisin.

Ce n'est pas un chunker "sémantique" au sens recherche de pointe (type
topic modeling), c'est une heuristique simple et explicable, volontairement
choisie pour rester lisible dans un portfolio — voir docs/DESIGN_CHOICES.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tiktoken

from src.config import CONFIG
from src.ingestion.loaders import RawDocument

_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def n_tokens(self) -> int:
        return len(_ENCODER.encode(self.text))


def _split_sentences(text: str) -> list[str]:
    # Découpage par phrase simple et robuste (évite une dépendance NLTK lourde
    # pour un besoin qui reste basique). Gère les abréviations courantes minimalement.
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ü])", text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(
    doc: RawDocument,
    embed_fn,
    cfg=CONFIG.chunking,
) -> list[Chunk]:
    """
    embed_fn : fonction (list[str]) -> np.ndarray de shape (n, dim), déjà normalisée.
    On l'injecte plutôt que d'importer l'embedder directement pour ne pas
    coupler le chunking à un provider d'embeddings précis (test unitaire facile).
    """
    sentences = _split_sentences(doc.text)
    if len(sentences) <= 1:
        return _finalize_chunks([doc.text], doc, cfg)

    embeddings = embed_fn(sentences)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # similarité entre phrase i et phrase i+1 (lissée sur `sentence_buffer` voisins)
    boundaries = [0]
    for i in range(len(sentences) - 1):
        window_a = embeddings[max(0, i - cfg.sentence_buffer): i + 1].mean(axis=0)
        window_b = embeddings[i + 1: i + 2 + cfg.sentence_buffer].mean(axis=0)
        sim = float(np.dot(window_a, window_b))
        if sim < cfg.semantic_similarity_threshold:
            boundaries.append(i + 1)
    boundaries.append(len(sentences))

    raw_groups = [
        " ".join(sentences[boundaries[i]: boundaries[i + 1]])
        for i in range(len(boundaries) - 1)
    ]

    return _finalize_chunks(raw_groups, doc, cfg)


def _finalize_chunks(groups: list[str], doc: RawDocument, cfg) -> list[Chunk]:
    """Applique les garde-fous min/max tokens : fusion des groupes trop
    courts, re-découpage des groupes trop longs par phrases."""
    merged: list[str] = []
    buffer = ""
    for g in groups:
        candidate = (buffer + " " + g).strip() if buffer else g
        if len(_ENCODER.encode(candidate)) < cfg.min_chunk_tokens:
            buffer = candidate
            continue
        merged.append(candidate)
        buffer = ""
    if buffer:
        if merged:
            merged[-1] = (merged[-1] + " " + buffer).strip()
        else:
            merged.append(buffer)

    final_texts: list[str] = []
    for g in merged:
        tokens = _ENCODER.encode(g)
        if len(tokens) <= cfg.max_chunk_tokens:
            final_texts.append(g)
            continue
        # re-split par phrases pour ne pas couper un mot en deux
        sub_sentences = _split_sentences(g)
        current = ""
        for s in sub_sentences:
            candidate = (current + " " + s).strip() if current else s
            if len(_ENCODER.encode(candidate)) > cfg.max_chunk_tokens and current:
                final_texts.append(current)
                current = s
            else:
                current = candidate
        if current:
            final_texts.append(current)

    chunks = []
    for i, text in enumerate(final_texts):
        chunks.append(
            Chunk(
                text=text,
                source=doc.source,
                metadata={**doc.metadata},
                chunk_id=f"{doc.source}::{doc.metadata.get('page', doc.metadata.get('section_title', i))}::{i}",
            )
        )
    return chunks


def chunk_documents(docs: list[RawDocument], embed_fn) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(semantic_chunk(doc, embed_fn))
    return all_chunks
