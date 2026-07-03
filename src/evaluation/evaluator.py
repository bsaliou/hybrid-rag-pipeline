"""
Évaluation quantifiée du pipeline RAG.

Deux modes :
  1. RAGAS (si `ragas` est installé et une clé LLM est configurée) : métriques
     "officielles" faithfulness / answer_relevancy / context_precision /
     context_recall, calculées par un LLM juge.
  2. Mini-framework maison (`SimpleEvaluator`) : dépendance zéro, ne nécessite
     aucun appel LLM supplémentaire pour les métriques de retrieval, et une
     version simplifiée "LLM-as-judge" pour la faithfulness. Utile pour
     itérer vite pendant le dev, ou pour montrer qu'on comprend ce que les
     métriques mesurent réellement plutôt que d'utiliser RAGAS en boîte noire.

Métriques calculées :
  - context_precision : parmi les chunks retournés, quelle fraction est
    effectivement pertinente pour répondre à la question (jugée par un LLM).
  - context_recall : parmi les faits nécessaires à la réponse de référence,
    quelle fraction est couverte par les chunks retournés.
  - faithfulness : quelle fraction des affirmations de la réponse générée
    est effectivement supportée par le contexte fourni (mesure l'hallucination).
  - answer_relevancy : la réponse générée répond-elle bien à la question posée
    (et pas à une question voisine) — mesurée par similarité embedding entre
    la question et des questions reconstruites à partir de la réponse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from src.embeddings.embedder import Embedder


@dataclass
class EvalExample:
    question: str
    ground_truth: str          # réponse de référence (rédigée à la main)
    generated_answer: str
    retrieved_contexts: list[str]


@dataclass
class EvalResult:
    question: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
        }


class SimpleEvaluator:
    """
    Mini-framework d'évaluation "maison". Utilise un LLM juge (Claude) pour
    les jugements sémantiques (faithfulness, pertinence des contextes), et
    des embeddings pour answer_relevancy (pas besoin de LLM, donc rapide et
    peu coûteux à faire tourner sur beaucoup d'exemples).
    """

    def __init__(self, llm_client=None, model: str = "claude-sonnet-4-6"):
        self.embedder = Embedder()
        self.model = model
        self._llm = llm_client
        if self._llm is None:
            from anthropic import Anthropic
            from src.config import CONFIG

            self._llm = Anthropic(api_key=CONFIG.generation.api_key)

    def evaluate(self, example: EvalExample) -> EvalResult:
        precision = self._context_precision(example)
        recall = self._context_recall(example)
        faithfulness = self._faithfulness(example)
        relevancy = self._answer_relevancy(example)
        return EvalResult(
            question=example.question,
            context_precision=precision,
            context_recall=recall,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
        )

    def evaluate_batch(self, examples: list[EvalExample]) -> list[EvalResult]:
        return [self.evaluate(ex) for ex in examples]

    # ---------- Métriques ----------

    def _context_precision(self, ex: EvalExample) -> float:
        if not ex.retrieved_contexts:
            return 0.0
        verdicts = [
            self._judge_relevance(ex.question, ctx) for ctx in ex.retrieved_contexts
        ]
        return sum(verdicts) / len(verdicts)

    def _context_recall(self, ex: EvalExample) -> float:
        """
        On décompose la réponse de référence en affirmations atomiques, puis
        on vérifie combien sont couvertes par au moins un des contextes
        retournés. Approxime le rappel sans nécessiter un jeu de contextes
        de référence annoté manuellement (coûteux à produire).
        """
        claims = self._extract_claims(ex.ground_truth)
        if not claims:
            return 1.0
        context_blob = "\n".join(ex.retrieved_contexts)
        covered = [self._judge_supported(claim, context_blob) for claim in claims]
        return sum(covered) / len(covered)

    def _faithfulness(self, ex: EvalExample) -> float:
        claims = self._extract_claims(ex.generated_answer)
        if not claims:
            return 1.0
        context_blob = "\n".join(ex.retrieved_contexts)
        supported = [self._judge_supported(claim, context_blob) for claim in claims]
        return sum(supported) / len(supported)

    def _answer_relevancy(self, ex: EvalExample) -> float:
        """
        Génère N questions hypothétiques à partir de la réponse, puis mesure
        la similarité cosinus moyenne avec la question originale. Si la
        réponse est pertinente, les questions reconstruites doivent être
        proches de la question posée (technique reprise de RAGAS).
        """
        reconstructed = self._generate_questions_from_answer(ex.generated_answer, n=3)
        if not reconstructed:
            return 0.0
        q_emb = self.embedder.embed_query(ex.question)
        r_embs = self.embedder.embed_documents(reconstructed)
        sims = r_embs @ q_emb
        return float(np.mean(sims))

    # ---------- Appels LLM juge ----------

    def _judge_relevance(self, question: str, context: str) -> bool:
        prompt = (
            f"Question : {question}\nPassage : {context}\n\n"
            "Ce passage contient-il une information utile pour répondre à la "
            'question ? Réponds uniquement par "oui" ou "non".'
        )
        resp = self._ask(prompt, max_tokens=5)
        return resp.strip().lower().startswith("oui")

    def _judge_supported(self, claim: str, context: str) -> bool:
        prompt = (
            f"Affirmation : {claim}\nContexte : {context}\n\n"
            "Cette affirmation est-elle directement supportée par le contexte "
            'ci-dessus (sans extrapolation) ? Réponds uniquement par "oui" ou "non".'
        )
        resp = self._ask(prompt, max_tokens=5)
        return resp.strip().lower().startswith("oui")

    def _extract_claims(self, text: str) -> list[str]:
        prompt = (
            f"Texte : {text}\n\n"
            "Décompose ce texte en affirmations factuelles atomiques (une "
            "affirmation simple par ligne, sans numérotation). "
            "Réponds au format JSON : une liste de strings."
        )
        resp = self._ask(prompt, max_tokens=512)
        return _parse_json_list(resp)

    def _generate_questions_from_answer(self, answer: str, n: int) -> list[str]:
        prompt = (
            f"Réponse : {answer}\n\n"
            f"Génère {n} questions différentes auxquelles cette réponse pourrait "
            "correspondre. Réponds au format JSON : une liste de strings."
        )
        resp = self._ask(prompt, max_tokens=256)
        return _parse_json_list(resp)

    def _ask(self, prompt: str, max_tokens: int) -> str:
        resp = self._llm.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


def _parse_json_list(text: str) -> list[str]:
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    try:
        parsed = json.loads(text[start: end + 1])
        return [str(x) for x in parsed]
    except json.JSONDecodeError:
        return [line.strip("- ").strip() for line in text.splitlines() if line.strip()]


def aggregate(results: list[EvalResult]) -> dict:
    if not results:
        return {}
    return {
        "n_examples": len(results),
        "avg_context_precision": round(np.mean([r.context_precision for r in results]), 3),
        "avg_context_recall": round(np.mean([r.context_recall for r in results]), 3),
        "avg_faithfulness": round(np.mean([r.faithfulness for r in results]), 3),
        "avg_answer_relevancy": round(np.mean([r.answer_relevancy for r in results]), 3),
    }
