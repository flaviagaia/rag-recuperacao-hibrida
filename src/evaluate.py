"""Avaliação: recall@k e MRR por tipo de consulta e no agregado."""

from __future__ import annotations

import json
from pathlib import Path


def load_queries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def rank_of_gold(ranked_ids: list[str], gold: str) -> int | None:
    for r, doc_id in enumerate(ranked_ids, start=1):
        if doc_id == gold:
            return r
    return None


def mrr(ranks: list[int | None]) -> float:
    return sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks)


def recall_at(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r and r <= k) / len(ranks)


def evaluate(retriever, queries: list[dict]) -> dict:
    """Retorna métricas por tipo de consulta e agregadas para um recuperador."""
    por_tipo: dict[str, list[int | None]] = {}
    todos: list[int | None] = []
    for item in queries:
        ranked = retriever.search(item["q"])
        r = rank_of_gold(ranked, item["gold"])
        por_tipo.setdefault(item["tipo"], []).append(r)
        todos.append(r)
    resumo = {tipo: {"mrr": mrr(rs), "recall@3": recall_at(rs, 3)}
              for tipo, rs in por_tipo.items()}
    resumo["agregado"] = {"mrr": mrr(todos), "recall@3": recall_at(todos, 3)}
    return resumo
