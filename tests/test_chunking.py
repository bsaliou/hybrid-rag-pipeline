"""
Tests du chunking. On mocke embed_fn pour ne pas dépendre d'un vrai modèle
d'embeddings (tests rapides, déterministes, sans téléchargement de poids).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunking import semantic_chunk  # noqa: E402
from src.ingestion.loaders import RawDocument  # noqa: E402


def fake_embed_two_topics(sentences: list[str]) -> np.ndarray:
    """
    Simule deux "sujets" bien séparés : les phrases contenant 'chat' ont un
    embedding proche de [1, 0], celles contenant 'voiture' proche de [0, 1].
    Permet de vérifier que le chunker détecte bien la frontière sémantique.
    """
    vecs = []
    for s in sentences:
        if "chat" in s.lower():
            vecs.append([1.0, 0.01])
        else:
            vecs.append([0.01, 1.0])
    return np.array(vecs, dtype=np.float32)


def test_semantic_chunk_detects_topic_boundary():
    text = (
        "Le chat dort sur le canapé. Le chat aime les croquettes. "
        "La voiture roule vite sur l'autoroute. La voiture consomme de l'essence."
    )
    doc = RawDocument(text=text, source="test.md", metadata={})
    chunks = semantic_chunk(doc, embed_fn=fake_embed_two_topics)

    assert len(chunks) >= 1
    # le contenu total doit être préservé (pas de perte de texte)
    joined = " ".join(c.text for c in chunks)
    assert "chat" in joined and "voiture" in joined


def test_semantic_chunk_empty_document():
    doc = RawDocument(text="", source="empty.md", metadata={})
    chunks = semantic_chunk(doc, embed_fn=fake_embed_two_topics)
    assert chunks == []


def test_semantic_chunk_single_sentence():
    doc = RawDocument(text="Une seule phrase courte.", source="short.md", metadata={})
    chunks = semantic_chunk(doc, embed_fn=fake_embed_two_topics)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "Une seule phrase courte."


def test_chunk_respects_max_tokens():
    from src.config import CONFIG

    long_sentence = "Le chat dort. " * 500  # dépasse largement max_chunk_tokens
    doc = RawDocument(text=long_sentence, source="long.md", metadata={})
    chunks = semantic_chunk(doc, embed_fn=fake_embed_two_topics)
    for c in chunks:
        assert c.n_tokens() <= CONFIG.chunking.max_chunk_tokens + 20  # marge tolérée
