"""Recuperação híbrida por Reciprocal Rank Fusion (RRF).

RRF combina vários rankings sem precisar normalizar scores de naturezas diferentes
(BM25 e cosseno não são comparáveis). Cada documento ganha pontos pelo INVERSO da
sua posição em cada lista: score(d) = Σ 1 / (k + rank_i(d)). Simples, robusto e
forte na prática (Cormack, Clarke & Buettcher, 2009).
"""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Funde várias listas ordenadas de ids em uma só, por RRF."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for pos, doc_id in enumerate(ranking):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda d: score[d], reverse=True)


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    """Híbrido por FUSÃO DE SCORES NORMALIZADOS (combinação convexa).

    BM25 e cosseno não são comparáveis em escala, então normalizamos cada um
    (min-max) por consulta e combinamos: score = w·léxico + (1-w)·denso. Com peso
    igual (w=0.5), o híbrido herda o que cada um tem de melhor. O peso é um
    hiperparâmetro (dialoga com o optuna-rag-tuning).

    Em acervos pequenos esta fusão supera o RRF (que achata as posições do topo);
    o RRF (`reciprocal_rank_fusion`) fica disponível para acervos grandes.
    """

    def __init__(self, lexical, dense, w: float = 0.5) -> None:
        self.lexical, self.dense, self.w = lexical, dense, w

    def search(self, query: str, top_k: int | None = None) -> list[str]:
        lex = _minmax(self.lexical.scores_by_id(query))
        den = _minmax(self.dense.scores_by_id(query))
        comb = {i: self.w * lex[i] + (1 - self.w) * den[i] for i in lex}
        ranked = sorted(comb, key=lambda i: comb[i], reverse=True)
        return ranked[:top_k] if top_k else ranked
