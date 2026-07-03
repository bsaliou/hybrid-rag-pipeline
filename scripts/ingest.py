"""
CLI d'indexation : ingère un fichier ou dossier dans la base vectorielle.

Usage :
    python scripts/ingest.py data/sample_docs
    python scripts/ingest.py path/vers/un/document.pdf
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Indexe des documents dans le RAG.")
    parser.add_argument("path", type=str, help="Fichier ou dossier à ingérer")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    n_chunks = pipeline.index_path(args.path)
    print(f"✅ {n_chunks} chunks indexés depuis '{args.path}'.")
    print(f"   Taille totale de la collection : {pipeline.store.count()} chunks.")


if __name__ == "__main__":
    main()
