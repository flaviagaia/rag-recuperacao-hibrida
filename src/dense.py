"""Recuperação densa (semântica).

Dois backends, mesma interface:
- 'lsa' (padrão): Indexação Semântica Latente (TF-IDF + TruncatedSVD). 100% offline,
  determinística, leve. Capta co-ocorrência/tema, indo além do termo exato, mas a
  semântica é mais fraca que a de um modelo neural.
- 'embeddings' (opcional): sentence-transformers, modelo multilíngue. Mais fiel à
  ideia de "recuperação densa por embeddings" da literatura (Karpukhin et al., 2020),
  mas exige baixar o modelo uma vez. Ativado só se a biblioteca estiver instalada.

A diferença de fundo para o BM25: o denso compara SIGNIFICADO, então acerta
paráfrase ("pedir permissão" ~ "consentimento") que o léxico não casa.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from corpus import Doc


class DenseRetriever:
    def __init__(self, docs: list[Doc], backend: str = "lsa", n_components: int = 16,
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.docs = docs
        self.backend = backend
        textos = [d.indexavel for d in docs]
        if backend == "lsa":
            n = min(n_components, max(2, len(docs) - 1))
            self._vec = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2))
            self._svd = make_pipeline(self._vec, TruncatedSVD(n_components=n, random_state=0),
                                      Normalizer(copy=False))
            self._mat = self._svd.fit_transform(textos)
        elif backend == "embeddings":
            from sentence_transformers import SentenceTransformer  # dep opcional
            self._model = SentenceTransformer(model_name)
            self._mat = self._model.encode(textos, normalize_embeddings=True)
        else:
            raise ValueError("backend deve ser 'lsa' ou 'embeddings'")

    def _embed_query(self, query: str) -> np.ndarray:
        if self.backend == "lsa":
            return self._svd.transform([query])
        return self._model.encode([query], normalize_embeddings=True)

    def scores_by_id(self, query: str) -> dict[str, float]:
        sims = cosine_similarity(self._embed_query(query), self._mat).ravel()
        return {self.docs[i].id: float(sims[i]) for i in range(len(self.docs))}

    def search(self, query: str, top_k: int | None = None) -> list[str]:
        sims = cosine_similarity(self._embed_query(query), self._mat).ravel()
        ranked = sims.argsort()[::-1]
        ids = [self.docs[i].id for i in ranked]
        return ids[:top_k] if top_k else ids
