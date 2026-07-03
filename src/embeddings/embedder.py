"""
Wrapper d'embeddings avec double backend :
  - local : sentence-transformers / BAAI-bge (gratuit, tourne sur CPU)
  - openai : text-embedding-3-large (payant, meilleure qualité multilingue)

Interface unifiée pour que le reste du pipeline (chunking, vectorstore,
retrieval) soit agnostique du provider choisi.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.config import CONFIG


class Embedder:
    def __init__(self, cfg=CONFIG.embedding):
        self.cfg = cfg
        self.provider = cfg.provider
        if self.provider == "local":
            self._model = _load_sentence_transformer(cfg.model_local)
        elif self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=cfg.openai_api_key)
        else:
            raise ValueError(f"Provider d'embeddings inconnu : {self.provider}")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embeddings pour des passages à indexer (pas de préfixe d'instruction)."""
        if self.provider == "local":
            vecs = self._model.encode(
                texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
            )
            return np.asarray(vecs, dtype=np.float32)
        return self._embed_openai(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Pour bge, les requêtes doivent être préfixées d'une instruction dédiée
        (asymétrie query/document) — c'est documenté par BAAI et améliore
        sensiblement le rappel en pratique.
        """
        if self.provider == "local":
            prefixed = self.cfg.bge_query_instruction + text
            vec = self._model.encode([prefixed], normalize_embeddings=True)
            return np.asarray(vec[0], dtype=np.float32)
        return self._embed_openai([text])[0]

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.cfg.model_openai, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        return vecs


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    # import différé + cache : le chargement du modèle (quelques centaines de Mo)
    # ne doit se faire qu'une fois par process, pas à chaque instanciation d'Embedder.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
