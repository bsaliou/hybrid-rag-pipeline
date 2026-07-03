"""
Fusion hybride dense + BM25 par Reciprocal Rank Fusion (RRF).

Pourquoi RRF plutôt qu'une moyenne pondérée des scores ?
Les scores denses (similarité cosinus, ~[0,1]) et les scores BM25
(non bornés, dépendants du corpus) ne sont pas sur la même échelle :
les combiner par pondération directe nécessite une calibration fragile.
RRF ne regarde que le *rang* de chaque document dans chaque liste, ce qui
le rend insensible à l'échelle des scores et robuste par défaut :

    RRF(d) = sum_over_retrievers( 1 / (k + rank(d)) )

C'est la méthode utilisée en production par Elastic, Weaviate, etc.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import CONFIG
from src.retrieval.bm25_retriever import BM25Result
from src.vectorstore.store import ScoredChunk


@dataclass
class FusedResult:
    chunk_id: str
    text: str
    source: str
    metadata: dict
    rrf_score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


def reciprocal_rank_fusion(
    dense_results: list[ScoredChunk],
    bm25_results: list[BM25Result],
    k: int = CONFIG.retrieval.rrf_k,
) -> list[FusedResult]:
    scores: dict[str, float] = {}
    info: dict[str, FusedResult] = {}

    for rank, r in enumerate(dense_results, start=1):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
        info[r.chunk_id] = FusedResult(
            chunk_id=r.chunk_id, text=r.text, source=r.source,
            metadata=r.metadata, rrf_score=0.0, dense_rank=rank,
        )

    for rank, r in enumerate(bm25_results, start=1):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
        if r.chunk_id in info:
            info[r.chunk_id].bm25_rank = rank
        else:
            info[r.chunk_id] = FusedResult(
                chunk_id=r.chunk_id, text=r.text, source=r.source,
                metadata=r.metadata, rrf_score=0.0, bm25_rank=rank,
            )

    for cid, s in scores.items():
        info[cid].rrf_score = s

    return sorted(info.values(), key=lambda x: x.rrf_score, reverse=True)
