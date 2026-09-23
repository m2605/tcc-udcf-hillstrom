# Scripts

Código que produziu os resultados documentados em `NOTAS_TECNICAS.md`.

**Ambiente:** WSL2 + Ubuntu 26.04 (g++ 15.2, cmake 4.2.3) para os modelos em C++;
Python 3.12 no Windows com `causalml`, `scikit-learn`, `pandas`, `numpy`, `scipy`,
`matplotlib` para o restante. Os scripts referenciam caminhos absolutos da máquina em que
foram executados — para rodar em outro lugar, ajustar as constantes no topo de cada um.

**Pré-requisito:** o repositório dos autores clonado e o `LBCF_RCT.zip` extraído, conforme
a seção 1 das notas técnicas.

---

## 01_dados

| arquivo | o que faz |
|---|---|
| `prep_visit.py` | baixa a base Hillstrom e monta os arquivos de entrada para os dois desfechos (`visit` e `conversion`), no formato que o código C++ espera |
| `mbcf_cv_prep.py` | monta as dobras de validação cruzada do MBCF, desfecho `conversion` |
| `mbcf_visit.py` | idem, desfecho `visit` |

## 02_modelos

**C++ — pontos de entrada.** Não são arquivos do algoritmo; substituem o `main.cpp` do
projeto dos autores, que o README deles designa como ponto de entrada.

| arquivo | o que faz |
|---|---|
| `main_ablacao.cpp` | UDCF e ablação. Recebe `min_node_size`, `imbalance_penalty` e `stabilize_splits` por linha de comando. Inclui blocos de diagnóstico somente-leitura (contagem de nós e `SplitFrequencyComputer`) |
| `main_mbcf_cv.cpp` | MBCF — treina num arquivo, prediz noutro. Usa `instrumental_trainer(0.0, false)` com instrumento igual ao tratamento, como o `main.cpp` original do CF_DT |

**Execução.**

| arquivo | o que faz |
|---|---|
| `passo5.sh` | experimento fatorial 2×2 na `conversion`: `min_node_size` × `imbalance_penalty` |
| `ablacao.sh` | UDCF vs. ablação (`stabilize_splits` true/false), `conversion` |
| `udcf_visit.sh` | UDCF e ablação no desfecho `visit` |
| `mbcf_cv_run.sh` · `mbcf_visit_run.sh` | MBCF com validação cruzada, cada desfecho |

**Baselines em Python.**

| arquivo | o que faz |
|---|---|
| `baseline_causalml.py` | Chi, ED e CTS com validação cruzada de 5 dobras, `conversion` |
| `meta_learners.py` | S-learner e T-learner, `conversion` |
| `baselines_visit.py` | os cinco acima, desfecho `visit` |

## 03_metricas

| arquivo | o que faz |
|---|---|
| `ordenacao_visit.py` | **Qini** (nulo por permutação, `normalize=False`) e **GATES** por quintil, `visit` |
| `politica_visit.py` | **valor da política** e curva de cobertura, `visit` |
| `avaliacao_politica.py` | idem, `conversion` |
| `metricas_biblioteca.py` | Qini, AUUC e RATE da biblioteca `causalml`, `conversion` |
| `tabela_final.py` | comparação por BLP e GATES entre todos os modelos, `conversion` |

## 04_diagnosticos

Sustentam os achados das seções 6.4 a 6.12 das notas.

| arquivo | o que estabelece |
|---|---|
| `diag_split_threshold.py` | a condição de split reduz a `‖s_esq‖² > imbalance_penalty`; mede a magnitude atingível na Hillstrom |
| `diag_authors_data.py` | a mesma medida na base dos autores — doze ordens de grandeza de diferença |
| **`escala_relabeling.py`** | **o mecanismo central**: o relabeling multi-braço normaliza por `(W'W)⁻¹ ~ 1/n`, o instrumental não |
| `diag_min_node_size2.py` | `min_node_size` é aplicado por braço dos dois lados; torna a amostra dos autores intreinável |
| `investigar_blp_negativo.py` | o artefato do `predict_oob` sobre floresta degenerada, com a assinatura de sinais opostos |
| `verificar_heterogeneidade*.py` | testes de interação diretamente nos dados brutos, sem modelo |
| `verificar_discriminativo.py` | separação entre braços por covariável — o alvo do critério Intra split |
| `familywise.py` | distribuição nula do menor p-valor entre os testes, com a correlação real entre modelos |
| `validar_blp.py` | valida a implementação do BLP em quatro cenários sintéticos de resposta conhecida |

## 05_figuras

| arquivo | o que gera |
|---|---|
| `curvas.py` | `curvas_qini.png` — desfecho `conversion` |
| `curvas_visit.py` | `curvas_qini_visit.png` — desfecho `visit` |

---

## Ordem de execução

```
01_dados/prep_visit.py
02_modelos/udcf_visit.sh              (e passo5.sh, ablacao.sh para conversion)
01_dados/mbcf_visit.py  ->  02_modelos/mbcf_visit_run.sh
02_modelos/baselines_visit.py         (~80 min)
03_metricas/ordenacao_visit.py
03_metricas/politica_visit.py
05_figuras/curvas_visit.py
```

Os diagnósticos em `04_diagnosticos/` são independentes e podem rodar a qualquer momento
após a preparação dos dados.
