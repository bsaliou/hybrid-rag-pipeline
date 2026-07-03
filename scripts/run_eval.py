"""
CLI d'évaluation : charge un jeu de questions/réponses de référence
(data/eval_dataset.json), fait tourner le pipeline dessus, calcule les
métriques et sauvegarde un rapport CSV + un résumé agrégé.

Usage :
    python scripts/run_eval.py
    python scripts/run_eval.py --use-ragas
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.evaluation.evaluator import EvalExample, SimpleEvaluator, aggregate  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval_results"


def main():
    parser = argparse.ArgumentParser(description="Évalue le pipeline RAG.")
    parser.add_argument("--use-ragas", action="store_true", help="Utilise RAGAS en plus du framework maison")
    args = parser.parse_args()

    with open(DATASET_PATH, encoding="utf-8") as f:
        golden_set = json.load(f)

    pipeline = RAGPipeline()
    examples = []
    for item in golden_set:
        answer, trace = pipeline.query(item["question"])
        examples.append(
            EvalExample(
                question=item["question"],
                ground_truth=item["ground_truth"],
                generated_answer=answer.answer,
                retrieved_contexts=[c.text for c in trace.final_chunks],
            )
        )

    evaluator = SimpleEvaluator()
    results = evaluator.evaluate_batch(examples)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame([r.as_dict() for r in results])
    df.to_csv(OUTPUT_DIR / "eval_report.csv", index=False)

    summary = aggregate(results)
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Résumé de l'évaluation ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nRapport détaillé : {OUTPUT_DIR / 'eval_report.csv'}")

    if args.use_ragas:
        from src.evaluation.ragas_eval import run_ragas_eval

        ragas_results = run_ragas_eval(examples)
        with open(OUTPUT_DIR / "ragas_report.json", "w", encoding="utf-8") as f:
            json.dump(ragas_results, f, indent=2, ensure_ascii=False)
        print(f"Rapport RAGAS : {OUTPUT_DIR / 'ragas_report.json'}")


if __name__ == "__main__":
    main()
