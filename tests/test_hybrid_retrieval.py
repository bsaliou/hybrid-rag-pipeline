import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.bm25_retriever import BM25Result  # noqa: E402
from src.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402
from src.vectorstore.store import ScoredChunk  # noqa: E402


def _dense(cid, score):
    return ScoredChunk(chunk_id=cid, text=f"text-{cid}", source="doc.md", metadata={}, score=score)


def _bm25(cid, score):
    return BM25Result(chunk_id=cid, text=f"text-{cid}", source="doc.md", metadata={}, score=score)


def test_rrf_favors_docs_ranked_high_in_both_lists():
    dense = [_dense("A", 0.9), _dense("B", 0.8), _dense("C", 0.7)]
    bm25 = [_bm25("B", 5.0), _bm25("A", 4.0), _bm25("D", 3.0)]

    fused = reciprocal_rank_fusion(dense, bm25, k=60)
    fused_ids = [f.chunk_id for f in fused]

    # A et B apparaissent en tête des deux listes -> doivent dominer le classement fusionné
    assert set(fused_ids[:2]) == {"A", "B"}


def test_rrf_includes_docs_present_in_only_one_list():
    dense = [_dense("A", 0.9)]
    bm25 = [_bm25("Z", 2.0)]
    fused = reciprocal_rank_fusion(dense, bm25, k=60)
    fused_ids = {f.chunk_id for f in fused}
    assert fused_ids == {"A", "Z"}


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], [], k=60) == []
