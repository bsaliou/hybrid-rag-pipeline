"""
Reranking : deuxième passe de scoring, plus coûteuse mais plus précise,
appliquée uniquement sur les ~20-40 candidats déjà remontés par le
retrieval hybride (pas sur tout le corpus — coût prohibitif sinon).

Pourquoi un reranker après un retrieval déjà hybride ?
Le retrieval (dense ou BM25) encode query et document *séparément*
(bi-encoder) : rapide mais imprécis, car il n'y a pas d'interaction directe
entre les deux textes au moment du scoring. Un cross-encoder prend la paire
(query, document) *ensemble* en entrée et produit un score d'interaction
fine — nettement plus précis, mais trop lent pour scanner tout le corpus.
Le pattern retrieve-then-rerank combine donc rappel (hybride) et précision
(cross-encoder) à coût maîtrisé.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.config import CONFIG
from src.retrieval.hybrid import FusedResult


@dataclass
class RerankedResult:
    chunk_id: str
    text: str
    source: str
    metadata: dict
    rerank_score: float


class Reranker:
    def __init__(self, cfg=CONFIG.reranker):
        self.cfg = cfg
        if cfg.provider == "local":
            self._model = _load_cross_encoder(cfg.model_local)
        elif cfg.provider == "cohere":
            import cohere

            self._client = cohere.Client(cfg.cohere_api_key)
        else:
            raise ValueError(f"Provider de reranking inconnu : {cfg.provider}")

    def rerank(
        self, query: str, candidates: list[FusedResult], top_k: int
    ) -> list[RerankedResult]:
        if not candidates:
            return []
        if self.cfg.provider == "local":
            return self._rerank_local(query, candidates, top_k)
        return self._rerank_cohere(query, candidates, top_k)

    def _rerank_local(
        self, query: str, candidates: list[FusedResult], top_k: int
    ) -> list[RerankedResult]:
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RerankedResult(
                chunk_id=c.chunk_id, text=c.text, source=c.source,
                metadata=c.metadata, rerank_score=float(s),
            )
            for c, s in ranked
        ]

    def _rerank_cohere(
        self, query: str, candidates: list[FusedResult], top_k: int
    ) -> list[RerankedResult]:
        docs = [c.text for c in candidates]
        resp = self._client.rerank(
            model=self.cfg.model_cohere, query=query, documents=docs, top_n=top_k
        )
        out = []
        for r in resp.results:
            c = candidates[r.index]
            out.append(
                RerankedResult(
                    chunk_id=c.chunk_id, text=c.text, source=c.source,
                    metadata=c.metadata, rerank_score=float(r.relevance_score),
                )
            )
        return out


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)
