"""
Interface Streamlit du pipeline RAG.

Lancer avec :
    streamlit run app/streamlit_app.py
"""
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="RAG Portfolio", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Chargement du pipeline (modèles d'embeddings/reranking)...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def main():
    st.title("🔎 RAG Portfolio — pipeline hybride avec reranking")
    st.caption(
        "Ingestion multi-format → chunking sémantique → retrieval hybride "
        "(dense + BM25) → reranking cross-encoder → génération avec citations."
    )

    pipeline = get_pipeline()

    with st.sidebar:
        st.header("📁 Indexer des documents")
        uploaded = st.file_uploader(
            "PDF, HTML ou Markdown", type=["pdf", "html", "htm", "md"], accept_multiple_files=True
        )
        if uploaded and st.button("Indexer"):
            with st.spinner("Ingestion + chunking + embedding en cours..."):
                total = 0
                for f in uploaded:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(f.name).suffix
                    ) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    total += pipeline.index_path(tmp_path)
                st.success(f"{total} chunks indexés.")

        st.divider()
        st.metric("Chunks dans la base", pipeline.store.count())

        st.divider()
        st.header("⚙️ Config active")
        from src.config import CONFIG

        st.text(f"Embeddings : {CONFIG.embedding.provider}")
        st.text(f"Reranker   : {CONFIG.reranker.provider}")
        st.text(f"Top-k dense: {CONFIG.retrieval.top_k_dense}")
        st.text(f"Top-k BM25 : {CONFIG.retrieval.top_k_bm25}")
        st.text(f"Top-k final: {CONFIG.retrieval.top_k_final}")

    query = st.text_input("Pose ta question", placeholder="Pourquoi utiliser le retrieval hybride ?")

    if query:
        if pipeline.store.count() == 0:
            st.warning("Aucun document indexé. Ajoute des documents dans la barre latérale, "
                       "ou lance `python scripts/ingest.py data/sample_docs` avant de démarrer l'app.")
            return

        with st.spinner("Retrieval hybride + reranking + génération..."):
            answer, trace = pipeline.query(query)

        st.subheader("Réponse")
        st.markdown(answer.answer)

        col1, col2, col3 = st.columns(3)
        col1.metric("Candidats denses", trace.dense_candidates)
        col2.metric("Candidats BM25", trace.bm25_candidates)
        col3.metric("Après fusion RRF", trace.fused_candidates)

        st.subheader("Passages utilisés (après reranking)")
        for i, c in enumerate(trace.final_chunks, start=1):
            loc = c.metadata.get("page") or c.metadata.get("section_title") or ""
            with st.expander(f"[{i}] {c.source} {f'— {loc}' if loc else ''} · score={c.rerank_score:.3f}"):
                st.write(c.text)


if __name__ == "__main__":
    main()
