"""
Pipeline RAG complet : ingestion -> chunking -> embedding -> indexation,
et requête -> retrieval hybride -> reranking -> génération.

Ce module est le point d'entrée utilisé à la fois par le CLI d'indexation,
l'app Streamlit et le harness d'évaluation — pour garantir que le chemin
"évalué" est exactement le chemin "servi en prod".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import CONFIG
from src.embeddings.embedder import Embedder
from src.generation.generator import GeneratedAnswer, generate_answer
from src.ingestion.chunking import Chunk, chunk_documents
from src.ingestion.loaders import load_directory, load_document
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker, RerankedResult
from src.vectorstore.store import VectorStore


@dataclass
class RetrievalTrace:
    """Trace complète du retrieval, utile pour le debug et l'évaluation."""
    query: str
    dense_candidates: int
    bm25_candidates: int
    fused_candidates: int
    final_chunks: list[RerankedResult]


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.reranker = Reranker()
        self._bm25: BM25Retriever | None = None
        self._all_chunks: list[Chunk] = []
        self._load_bm25_from_store()

    def _load_bm25_from_store(self) -> None:
        """
        BM25 n'est pas persisté par Chroma (c'est un index en mémoire sur le
        corpus). Au démarrage du process, on reconstruit le corpus BM25 à
        partir des documents déjà présents dans la base vectorielle, pour que
        le retrieval hybride fonctionne même sans ré-ingestion dans la même
        session.
        """
        if self.store.count() == 0:
            return
        raw = self.store.collection.get(include=["documents", "metadatas"])
        chunks = [
            Chunk(text=doc, source=meta.get("source", ""), metadata=meta, chunk_id=cid)
            for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]
        self._all_chunks = chunks
        self._bm25 = BM25Retriever(chunks)

    # ---------- Indexation ----------

    def index_path(self, path: str | Path) -> int:
        """Ingère un fichier ou un dossier entier, chunk, embed et indexe."""
        path = Path(path)
        raw_docs = load_document(path) if path.is_file() else load_directory(path)

        chunks = chunk_documents(raw_docs, embed_fn=self.embedder.embed_documents)
        if not chunks:
            return 0

        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add_chunks(chunks, embeddings)

        self._all_chunks.extend(chunks)
        self._bm25 = BM25Retriever(self._all_chunks)  # réindexation BM25 (corpus en mémoire)
        return len(chunks)

    # ---------- Requête ----------

    def retrieve(self, query: str) -> RetrievalTrace:
        cfg = CONFIG.retrieval

        query_emb = self.embedder.embed_query(query)
        dense_results = self.store.query(query_emb, top_k=cfg.top_k_dense)

        bm25_results = self._bm25.query(query, top_k=cfg.top_k_bm25) if self._bm25 else []

        fused = reciprocal_rank_fusion(dense_results, bm25_results, k=cfg.rrf_k)
        reranked = self.reranker.rerank(query, fused, top_k=cfg.top_k_final)

        return RetrievalTrace(
            query=query,
            dense_candidates=len(dense_results),
            bm25_candidates=len(bm25_results),
            fused_candidates=len(fused),
            final_chunks=reranked,
        )

    def query(self, query: str) -> tuple[GeneratedAnswer, RetrievalTrace]:
        trace = self.retrieve(query)
        answer = generate_answer(query, trace.final_chunks)
        return answer, trace
