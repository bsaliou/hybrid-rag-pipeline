"""
Wrapper autour de ChromaDB (persistant, local, zéro infra à gérer — pas
besoin de faire tourner un serveur Qdrant en Docker pour un portfolio).
Le code est écrit pour que passer à Qdrant plus tard soit un simple
remplacement de cette classe (même interface add / query).
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
import numpy as np

from src.config import CONFIG
from src.ingestion.chunking import Chunk


@dataclass
class ScoredChunk:
    chunk_id: str
    text: str
    source: str
    metadata: dict
    score: float  # score de similarité (plus haut = plus pertinent)


class VectorStore:
    def __init__(self, cfg=CONFIG.vectorstore):
        self.client = chromadb.PersistentClient(path=cfg.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, **_flatten(c.metadata)} for c in chunks],
        )

    def query(self, query_embedding: np.ndarray, top_k: int) -> list[ScoredChunk]:
        res = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[ScoredChunk] = []
        for i in range(len(res["ids"][0])):
            distance = res["distances"][0][i]  # distance cosinus (0 = identique)
            score = 1.0 - distance
            out.append(
                ScoredChunk(
                    chunk_id=res["ids"][0][i],
                    text=res["documents"][0][i],
                    source=res["metadatas"][0][i].get("source", ""),
                    metadata=res["metadatas"][0][i],
                    score=score,
                )
            )
        return out

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name, metadata={"hnsw:space": "cosine"}
        )


def _flatten(metadata: dict) -> dict:
    # Chroma n'accepte que str/int/float/bool dans les métadonnées
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
