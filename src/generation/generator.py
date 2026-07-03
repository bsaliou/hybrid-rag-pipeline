"""
Génération de la réponse finale à partir des chunks rerankés.

Le prompt force explicitement :
  - la citation des sources utilisées (numéro de chunk),
  - le refus de répondre si le contexte ne contient pas l'information
    (réduit les hallucinations, mesurable ensuite via la métrique
    "faithfulness" du module d'évaluation).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import CONFIG
from src.retrieval.reranker import RerankedResult

SYSTEM_PROMPT = """Tu es un assistant qui répond STRICTEMENT à partir du contexte fourni.

Règles :
1. N'utilise que les informations présentes dans le contexte ci-dessous.
2. Si le contexte ne permet pas de répondre, dis-le explicitement plutôt que d'inventer.
3. Cite tes sources en indiquant le numéro du passage entre crochets, ex: [1], [2].
4. Sois concis et direct."""


@dataclass
class GeneratedAnswer:
    answer: str
    used_chunks: list[RerankedResult]


def build_context_block(chunks: list[RerankedResult]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        loc = c.metadata.get("page") or c.metadata.get("section_title") or ""
        parts.append(f"[{i}] (source: {c.source}{f', {loc}' if loc else ''})\n{c.text}")
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: list[RerankedResult], cfg=CONFIG.generation) -> GeneratedAnswer:
    from anthropic import Anthropic

    client = Anthropic(api_key=cfg.api_key)
    context = build_context_block(chunks)

    user_message = f"Contexte :\n{context}\n\nQuestion : {query}"

    resp = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return GeneratedAnswer(answer=text, used_chunks=chunks)
