# Fundamentação científica — H1 (recuperação léxica, densa e híbrida)

Estudo de literatura que sustenta a hipótese H1 e este experimento. Serve tanto ao
repositório quanto à revisão da dissertação. Citações no estilo autor-data.

## A hipótese
**H1.** A recuperação híbrida, que combina busca léxica (BM25) e densa (embeddings)
por fusão, supera as estratégias puramente léxica e puramente densa na recuperação
de dispositivos normativos pertinentes.

## 1. Recuperação léxica (BM25)
A família probabilística de relevância e o BM25 (Robertson & Zaragoza, 2009)
pontuam documentos pela sobreposição de **termos**, com saturação de frequência
(k1) e normalização por comprimento (b). Forças bem documentadas: precisão em
**correspondência exata** (números de dispositivos, siglas, anos, expressões
literais), robustez sem treino e interpretabilidade. Fraqueza central: é cego a
sinônimos e a paráfrase, o problema clássico de *vocabulary mismatch* na
Recuperação de Informação. Em domínio jurídico, onde a consulta cita "Art. 7º, II",
o número da resolução e a sigla do programa, essa força lexical é decisiva.

## 2. Recuperação densa (embeddings)
A recuperação densa representa consulta e documento como vetores e compara por
proximidade semântica. O marco é o Dense Passage Retrieval (Karpukhin et al.,
2020), que mostrou ganhos sobre BM25 em perguntas de domínio aberto ao capturar
**significado** além do termo exato, mitigando o *vocabulary mismatch*. A força é
exatamente a paráfrase ("pedir autorização" ≈ "consentimento"); a fraqueza é
errar em consultas que dependem de um identificador literal raro, que o vetor
semântico tende a diluir.

## 3. Híbrido e fusão de rankings
Como léxico e denso falham em situações **complementares**, combiná-los é natural.
A Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009) funde rankings sem
exigir scores comparáveis, somando o inverso das posições; é simples e forte em
acervos grandes. Em acervos pequenos, a fusão por **score normalizado** (combinação
convexa léxico/denso) costuma ser mais estável, pois o RRF achata as diferenças do
topo. O peso da combinação é um hiperparâmetro (conecta-se à otimização de
hiperparâmetros de RAG).

## 4. Evidência no domínio jurídico em português
A literatura recente em português jurídico mostra que **a superioridade não é
absoluta** e depende da consulta e do corpus. Ferreira, Braz & Ladeira (2024)
compararam busca semântica e léxica e não encontraram dominância universal de uma
sobre a outra. No conjunto JurisTCU (Fernandes et al., 2026), tanto estratégias
léxicas (BM25 com expansão) quanto densas obtêm ganhos expressivos, conforme o tipo
de consulta. Modelos e arquiteturas jurídicas como o Juru (Malaquias Junior et al.,
2024), o CBR-RAG (Wiratunga et al., 2024) e o benchmark LexRAG (Li et al., 2025)
consolidam o RAG no direito, mas concentram-se em jurisprudência e consulta, não no
cruzamento norma-execução. Esses achados motivam tratar a estratégia de recuperação
como **variável experimental** e investigar a fusão.

## 5. A lacuna e a contribuição
Não se identificam, no recorte recente, estudos que comparem sistematicamente
léxico, denso e híbrido **na legislação de programas públicos brasileiros**, com
consultas separadas por tipo (termo exato vs paráfrase). Este experimento
operacionaliza H1 nesse recorte: mede recall@k e MRR por tipo e no agregado, isolando
quando cada estratégia vence.

## 6. O que o experimento mostra (resultado offline, LGPD arts. 6º e 7º)
| recuperador | termo exato (MRR) | paráfrase (MRR) | agregado (MRR) |
| ----------- | ----------------- | --------------- | -------------- |
| BM25 (léxico) | 0.92 | 0.56 | 0.74 |
| Denso (LSA) | 0.88 | 0.62 | 0.75 |
| Híbrido (w=0.5) | **1.00** | 0.53 | **0.76** |

Coerente com a literatura: o léxico lidera o termo exato, o denso lidera a
paráfrase, e o híbrido tem o melhor desempenho agregado e o melhor em termo exato.

## 7. Limitações honestas
Corpus pequeno (dois artigos) e backend denso padrão por LSA (semântica latente),
mais fraco que embeddings neurais; o efeito da paráfrase é maior com o modo
`embeddings`. A fusão usada é por score normalizado (RRF disponível para acervos
grandes). São limitações de escopo do exemplo didático, não da hipótese.

## Referências
- ROBERTSON, S.; ZARAGOZA, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in IR*.
- KARPUKHIN, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP*.
- CORMACK, G.; CLARKE, C.; BUETTCHER, S. (2009). Reciprocal Rank Fusion. *SIGIR*.
- FERREIRA; BRAZ; LADEIRA (2024). Busca semântica vs léxica no domínio jurídico em português.
- FERNANDES et al. (2026). JurisTCU.
- MALAQUIAS JUNIOR et al. (2024). Juru. · WIRATUNGA et al. (2024). CBR-RAG. · LI et al. (2025). LexRAG.
- LEWIS, P. et al. (2020). Retrieval-Augmented Generation. *NeurIPS*. · GAO, Y. et al. (2024). RAG: A Survey.
