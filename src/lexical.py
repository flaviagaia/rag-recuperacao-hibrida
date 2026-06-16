"""Recuperação léxica: BM25 (Okapi) implementado do zero.

BM25 pontua pela sobreposição de TERMOS, com saturação de frequência (k1) e
normalização por tamanho do documento (b). É forte quando a consulta traz o termo
exato: número de artigo, sigla, expressão literal. É cego a sinônimos e paráfrase.

Robertson & Zaragoza (2009).
"""

from __future__ import annotations

import math
from collections import Counter

from corpus import Doc, tokenize


class BM25:
    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self.toks = [tokenize(d.indexavel) for d in docs]
        self.len = [len(t) for t in self.toks]
        self.avglen = sum(self.len) / len(self.len)
        self.tf = [Counter(t) for t in self.toks]
        # idf de cada termo (forma do BM25, com suavização)
        df: Counter = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - dfw + 0.5) / (dfw + 0.5)) for w, dfw in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                if w in tf:
                    idf = self.idf.get(w, 0.0)
                    num = tf[w] * (self.k1 + 1)
                    den = tf[w] + self.k1 * (1 - self.b + self.b * self.len[i] / self.avglen)
                    s += idf * num / den
            out.append(s)
        return out

    def scores_by_id(self, query: str) -> dict[str, float]:
        return {self.docs[i].id: s for i, s in enumerate(self.scores(query))}

    def search(self, query: str, top_k: int | None = None) -> list[str]:
        sc = self.scores(query)
        ranked = sorted(range(len(self.docs)), key=lambda i: sc[i], reverse=True)
        ids = [self.docs[i].id for i in ranked]
        return ids[:top_k] if top_k else ids
