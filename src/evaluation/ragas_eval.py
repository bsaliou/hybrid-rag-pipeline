"""
Wrapper optionnel pour évaluer avec la librairie RAGAS officielle, en plus
du SimpleEvaluator maison. Utile pour comparer les deux et justifier,
dans docs/DESIGN_CHOICES.md, la cohérence entre une implémentation "boîte
noire" (RAGAS) et une implémentation comprise de bout en bout (maison).

Nécessite : pip install ragas datasets + une clé LLM configurée.
"""
from __future__ import annotations

from src.evaluation.evaluator import EvalExample


def run_ragas_eval(examples: list[EvalExample]) -> dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict(
        {
            "question": [e.question for e in examples],
            "answer": [e.generated_answer for e in examples],
            "contexts": [e.retrieved_contexts for e in examples],
            "ground_truth": [e.ground_truth for e in examples],
        }
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result.to_pandas().to_dict(orient="records")
