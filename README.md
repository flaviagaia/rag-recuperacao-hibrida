# BM25 vs Denso vs Híbrido (recuperação para RAG)

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · scikit-learn · 100% offline (LSA), sem API key · MIT License

Dados públicos: LGPD (Lei nº 13.709/2018), arts. 6º e 7º.

> **Em uma frase:** "léxico ou semântico?" é a pergunta errada. Em texto normativo,
> o léxico (BM25) ganha nas consultas de termo exato (número do artigo, sigla), o
> denso ganha nas de paráfrase, e o **híbrido** (fusão dos dois) tem o melhor
> desempenho agregado. No experimento: híbrido com **MRR 0.76** vs 0.74 (BM25) e
> 0.75 (denso), e **1.00** no termo exato.

---

## 🇧🇷 Português

### O problema (e a resposta à crítica comum)
"Comparar BM25 com embeddings é coisa antiga." Não é, quando o tipo de consulta
mudou. A questão não é qual é melhor em abstrato, e sim **quando cada um ganha**:

- consultas de **termo exato** ("Art. 7º, inciso II", sigla, ano) → o léxico (BM25)
  casa o token raro que o denso dilui;
- consultas por **paráfrase** ("posso usar dados sem pedir autorização?") → o denso
  capta o significado que o léxico não casa;
- o **híbrido** herda as duas forças.

Fundamentação completa e referências em [`ESTUDO_CIENTIFICO_H1.md`](ESTUDO_CIENTIFICO_H1.md).

### Como funciona (o técnico)
Três recuperadores sobre os mesmos dispositivos da LGPD:

```
BM25 (léxico)   : Okapi BM25 do zero (k1=1.5, b=0.75). Sobreposição de termos.
Denso (semântico): LSA (TF-IDF + TruncatedSVD) por padrão, offline;
                   sentence-transformers opcional (modo 'embeddings').
Híbrido         : fusão por score normalizado -> w·minmax(léxico) + (1-w)·minmax(denso)
                  (RRF de Cormack et al. também incluído, melhor em acervos grandes)
```

O peso `w` é um hiperparâmetro (dialoga com o `optuna-rag-tuning`).

### Resultado real deste repositório
12 consultas, rotuladas em **termo exato** e **paráfrase**, com dispositivo-gabarito:

| recuperador | termo exato (MRR/r@3) | paráfrase (MRR/r@3) | agregado (MRR/r@3) |
| ----------- | --------------------- | ------------------- | ------------------ |
| BM25 (léxico) | 0.92 / 1.00 | 0.56 / 0.67 | 0.74 / 0.83 |
| Denso (LSA) | 0.88 / 0.83 | **0.62** / **0.83** | 0.75 / 0.83 |
| **Híbrido (w=0.5)** | **1.00** / 1.00 | 0.53 / 0.67 | **0.76 / 0.83** |

O léxico lidera o termo exato; o denso lidera a paráfrase; o híbrido tem o melhor
agregado e o melhor termo exato. (Com peso ajustado para o denso, w=0.3, o híbrido
sobe a 0.81 agregado e empata o denso na paráfrase — daí o peso ser um hiperparâmetro.)

### Como explicar em 30 segundos
BM25 é o que procura a palavra exata; o denso é o que entende o sentido. Pergunta
com número de artigo, o exato ganha; pergunta com suas palavras, o sentido ganha.
O híbrido não te obriga a escolher: ele soma os dois e fica bom nas duas situações.

### Execução
```
pip install -r requirements.txt
python src/demo.py                # denso por LSA (offline, padrão)
python src/demo.py embeddings     # denso por sentence-transformers (baixa o modelo 1x)
pytest tests/ -v                  # 7 testes
```

### Estrutura
```
data/lei_lgpd.json   # dispositivos (arts. 6º e 7º)
data/consultas.json  # consultas rotuladas (termo_exato / parafrase) + gabarito
src/lexical.py       # BM25 do zero
src/dense.py         # denso: LSA (padrão) + embeddings (opcional)
src/fusion.py        # fusão por score normalizado + RRF
src/evaluate.py      # recall@k e MRR por tipo e agregado
src/demo.py          # o bake-off
ESTUDO_CIENTIFICO_H1.md  # revisão de literatura que sustenta a hipótese
```

### Limitações honestas
Corpus pequeno (dois artigos) e denso padrão por LSA (semântica latente, mais fraca
que embeddings neurais); o ganho da paráfrase é maior no modo `embeddings`. O objetivo
é isolar **quando** cada estratégia vence, fiel ao que a literatura encontra.

### Referências científicas (crédito aos autores)
- Robertson & Zaragoza (2009), BM25. · Karpukhin et al. (2020), Dense Passage Retrieval, EMNLP.
- Cormack, Clarke & Buettcher (2009), Reciprocal Rank Fusion, SIGIR.
- Ferreira, Braz & Ladeira (2024); Fernandes et al. (2026, JurisTCU) — léxico vs denso em PT jurídico.
- Lewis et al. (2020), RAG; Gao et al. (2024), survey. Detalhes em `ESTUDO_CIENTIFICO_H1.md`.

Reimplementação didática e offline; crédito às autoras e aos autores originais.

---

## 🇺🇸 English

**In one line:** "lexical or semantic?" is the wrong question. On normative text,
lexical (BM25) wins exact-term queries (article number, acronym), dense wins
paraphrase, and the **hybrid** has the best aggregate. Here: hybrid MRR 0.76 vs 0.74
(BM25) and 0.75 (dense), and 1.00 on exact-term.

### How it works
Three retrievers over the same LGPD provisions: BM25 from scratch; dense via LSA
(TF-IDF + TruncatedSVD, offline) with optional sentence-transformers; hybrid via
normalized-score fusion (RRF also included). Queries are labeled exact-term vs
paraphrase. The weight `w` is a hyperparameter (ties to `optuna-rag-tuning`).

### Real result
BM25 leads exact-term (0.92 MRR), dense leads paraphrase (0.62), hybrid leads
aggregate (0.76) and exact-term (1.00). Consistent with the literature that no single
method dominates (Ferreira, Braz & Ladeira, 2024).

### Running
```
pip install -r requirements.txt
python src/demo.py
pytest tests/ -v          # 7 tests
```

### References
Robertson & Zaragoza (2009); Karpukhin et al. (2020); Cormack et al. (2009);
Lewis et al. (2020); Gao et al. (2024). See `ESTUDO_CIENTIFICO_H1.md`.

---

Part of my LinkedIn series on RAG efficiency → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
