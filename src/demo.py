"""Bake-off: BM25 (léxico) vs denso vs híbrido (RRF), por tipo de consulta.

    python src/demo.py                 # denso por LSA (offline, padrão)
    python src/demo.py embeddings      # denso por sentence-transformers (baixa o modelo)
"""

from __future__ import annotations

import sys
from pathlib import Path

from corpus import load_docs
from dense import DenseRetriever
from evaluate import evaluate, load_queries
from fusion import HybridRetriever
from lexical import BM25

ROOT = Path(__file__).parent.parent


def linha(nome: str, resumo: dict) -> str:
    def cel(t):
        return f"{resumo[t]['mrr']:.2f}/{resumo[t]['recall@3']:.2f}"
    return (f"{nome:<16}{cel('termo_exato'):>16}{cel('parafrase'):>16}"
            f"{cel('agregado'):>16}")


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "lsa"
    docs = load_docs(ROOT / "data" / "lei_lgpd.json")
    queries = load_queries(ROOT / "data" / "consultas.json")

    bm25 = BM25(docs)
    denso = DenseRetriever(docs, backend=backend)
    hibrido = HybridRetriever(bm25, denso, w=0.5)

    print("=" * 64)
    print(f"BM25 vs denso ({backend}) vs híbrido (fusão de scores) — MRR/recall@3")
    print("=" * 64)
    print(f"{'recuperador':<16}{'termo exato':>16}{'paráfrase':>16}{'agregado':>16}")
    print("-" * 64)
    print(linha("BM25 (léxico)", evaluate(bm25, queries)))
    print(linha(f"denso ({backend})", evaluate(denso, queries)))
    print(linha("híbrido (w=0.5)", evaluate(hibrido, queries)))
    print("\n  Esperado: léxico forte em termo exato, denso forte em paráfrase,")
    print("  híbrido o mais robusto no agregado (não desaba em nenhum tipo).")


if __name__ == "__main__":
    main()
