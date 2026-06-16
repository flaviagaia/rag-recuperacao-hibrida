"""H1 vira invariante: léxico forte em termo exato, denso em paráfrase, híbrido no topo.

Tudo com o backend LSA (offline, determinístico), o padrão reprodutível do repo.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from corpus import load_docs  # noqa: E402
from dense import DenseRetriever  # noqa: E402
from evaluate import evaluate, load_queries  # noqa: E402
from fusion import HybridRetriever, reciprocal_rank_fusion  # noqa: E402
from lexical import BM25  # noqa: E402

docs = load_docs(ROOT / "data" / "lei_lgpd.json")
queries = load_queries(ROOT / "data" / "consultas.json")
bm25 = BM25(docs)
denso = DenseRetriever(docs, backend="lsa")
hibrido = HybridRetriever(bm25, denso, w=0.5)

E_BM, E_DN, E_HB = evaluate(bm25, queries), evaluate(denso, queries), evaluate(hibrido, queries)


def test_lexico_forte_em_termo_exato():
    """BM25 não perde para o denso nas consultas de termo exato."""
    assert E_BM["termo_exato"]["mrr"] >= E_DN["termo_exato"]["mrr"]
    assert E_BM["termo_exato"]["mrr"] >= 0.9


def test_denso_forte_em_parafrase():
    """O denso supera o BM25 nas consultas por paráfrase."""
    assert E_DN["parafrase"]["mrr"] > E_BM["parafrase"]["mrr"]


def test_hibrido_lidera_o_agregado():
    """O híbrido tem o melhor MRR agregado e não fica abaixo no recall@3."""
    assert E_HB["agregado"]["mrr"] >= E_BM["agregado"]["mrr"]
    assert E_HB["agregado"]["mrr"] >= E_DN["agregado"]["mrr"]
    assert E_HB["agregado"]["recall@3"] >= max(E_BM["agregado"]["recall@3"],
                                                E_DN["agregado"]["recall@3"])


def test_hibrido_herda_a_forca_lexica():
    """O híbrido mantém o desempenho de termo exato (herda o BM25)."""
    assert E_HB["termo_exato"]["mrr"] >= E_BM["termo_exato"]["mrr"]


def test_bm25_acerta_rotulo_exato():
    """Consulta só com o rótulo ('Art. 7º, inciso X') retorna o dispositivo certo no topo."""
    assert bm25.search("Art. 7º, inciso X", top_k=1) == ["art7_X"]


def test_rrf_unitario():
    """RRF combina rankings premiando quem aparece no topo das duas listas."""
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert fused[0] in ("a", "b")  # a e b lideram as duas listas
    assert fused.index("a") < fused.index("c")


def test_determinismo():
    assert evaluate(hibrido, queries) == E_HB
