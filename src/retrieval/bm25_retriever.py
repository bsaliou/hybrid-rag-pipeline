"""
Retriever BM25 (lexical / sparse), complémentaire du retrieval dense.

Pourquoi garder BM25 alors qu'on a des embeddings ?
Voir docs/DESIGN_CHOICES.md pour la justification complète — en résumé :
BM25 excelle sur les termes exacts (identifiants, acronymes, noms propres,
numéros de version) que les embeddings denses ont tendance à "lisser" par
similarité sémantique, au détriment de la précision lexicale.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.ingestion.chunking import Chunk


@dataclass
class BM25Result:
    chunk_id: str
    text: str
    source: str
    metadata: dict
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9À-ÿ]+", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._corpus_tokens = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if chunks else None

    def query(self, query: str, top_k: int) -> list[BM25Result]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            BM25Result(
                chunk_id=self.chunks[i].chunk_id,
                text=self.chunks[i].text,
                source=self.chunks[i].source,
                metadata=self.chunks[i].metadata,
                score=float(scores[i]),
            )
            for i in ranked_idx
            if scores[i] > 0
        ]
