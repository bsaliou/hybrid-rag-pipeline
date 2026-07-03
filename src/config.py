"""
Configuration centralisée du pipeline RAG.

Toutes les valeurs sont lues depuis les variables d'environnement (.env),
avec des valeurs par défaut qui font tourner le pipeline entièrement en
local (aucune clé API requise) pour l'ingestion + le retrieval.
Seule la génération finale de réponse nécessite une clé LLM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = os.getenv("EMBEDDING_PROVIDER", "local")  # "local" | "openai"
    model_local: str = os.getenv("EMBEDDING_MODEL_LOCAL", "BAAI/bge-large-en-v1.5")
    model_openai: str = os.getenv("EMBEDDING_MODEL_OPENAI", "text-embedding-3-large")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    # bge attend un préfixe d'instruction pour les *queries* (pas pour les documents)
    bge_query_instruction: str = (
        "Represent this sentence for searching relevant passages: "
    )


@dataclass(frozen=True)
class RerankerConfig:
    provider: str = os.getenv("RERANKER_PROVIDER", "local")  # "local" | "cohere"
    model_local: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_cohere: str = "rerank-english-v3.0"
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")


@dataclass(frozen=True)
class VectorStoreConfig:
    persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", str(ROOT_DIR / "chroma_db"))
    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_portfolio")


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_dense: int = int(os.getenv("TOP_K_DENSE", 20))
    top_k_bm25: int = int(os.getenv("TOP_K_BM25", 20))
    top_k_final: int = int(os.getenv("TOP_K_FINAL_AFTER_RERANK", 5))
    rrf_k: int = int(os.getenv("RRF_K", 60))  # constante de smoothing du Reciprocal Rank Fusion


@dataclass(frozen=True)
class ChunkingConfig:
    # Chunking "sémantique" : on découpe d'abord par structure (titres, paragraphes),
    # puis on fusionne/scinde en fonction d'un seuil de similarité entre phrases
    # consécutives, avec un fallback taille fixe en garde-fou.
    min_chunk_tokens: int = 128
    max_chunk_tokens: int = 512
    semantic_similarity_threshold: float = 0.55
    sentence_buffer: int = 1  # nb de phrases voisines regardées pour le lissage


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = "anthropic"
    model: str = os.getenv("GENERATION_MODEL", "claude-sonnet-4-6")
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    max_tokens: int = 1024
    temperature: float = 0.1


@dataclass(frozen=True)
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


CONFIG = Config()
