# Notas técnicas — UDCF/LBCF aplicado à Hillstrom

Material de referência sobre o repositório dos autores, os modelos disponíveis e os
achados da aplicação à base Hillstrom. **Não é texto do TCC** — é documentação de apoio,
para consulta e para responder a perguntas de banca.

Cada afirmação aqui foi verificada lendo o código ou executando-o. As que **não** foram
verificadas estão marcadas explicitamente na seção 7.

---

## Índice

1. [Estrutura do repositório dos autores](#1-estrutura-do-repositório-dos-autores)
2. [O modelo UDCF: artigo × código](#2-o-modelo-udcf-artigo--código)
3. [Formato de entrada exigido pelo código](#3-formato-de-entrada-exigido-pelo-código)
4. [As baselines do artigo](#4-as-baselines-do-artigo)
5. [Como os autores avaliam os modelos](#5-como-os-autores-avaliam-os-modelos)
6. [Achados na aplicação à Hillstrom](#6-achados-na-aplicação-à-hillstrom)
7. [Estado de verificação](#7-estado-de-verificação)
8. [Ambiente de execução](#8-ambiente-de-execução)

---

## 1. Estrutura do repositório dos autores

Repositório: <https://github.com/www2022paper/WWW-2022-PAPER-SUPPLEMENTARY-MATERIALS>

```
Code/
  Data_generation/     geração dos dados sintéticos (R)
  Model/
    LBCF/              MÉTODO PROPOSTO (UDCF + DGB)
    CF_DT/             baseline — MBCF (K florestas binárias) + otim. determinística
    CT_ST/             baseline — causal tree + otim. estocástica (R)
    Chi_ED_CTS/        baselines — Chi, Euclidean Distance, CTS (CausalML)
  Evaluation/          scripts de avaliação (ITE e PMG)
Data/
  RCT_data/            amostra real (2.000 linhas, criptografada)
  Synthetic_data/      dados simulados (80.000 linhas × 4 níveis de incerteza)
Images/
```

`Code/Model/README.md` é explícito: **LBCF é o método proposto**; os outros três são
baselines.

### Armadilha: código duplicado na pasta de uma baseline

A pasta `Code/Model/CF_DT/` contém **também** `UDCF_RCT.zip` e `UDCF_Synthetic.zip`.

```
md5  bed42eac3acdca1ba409e7549609a2d3   Code/Model/LBCF/LBCF_RCT.zip
md5  bed42eac3acdca1ba409e7549609a2d3   Code/Model/CF_DT/UDCF_RCT.zip
```

São **byte a byte o mesmo arquivo**. O motivo: o `main.cpp` do MBCF (dentro de
`CF_DT_RCT.zip`) contém o código do UDCF **comentado no topo** — os autores reaproveitaram
o mesmo projeto C++ para os dois modelos e deixaram os zips de ambos na pasta.

**Consequência prática:** quem procurar "UDCF" no repositório pode acabar na pasta da
baseline. O caminho correto, segundo `Code/Model/LBCF/README.md`, é
`Code/Model/LBCF/LBCF_RCT.zip`.

### Cadeia de READMEs a seguir

| arquivo | o que informa |
|---|---|
| `README.md` (raiz) | instruções de reprodução das Seções 5.1 e 5.2; detalhes de implementação |
| `Code/Model/README.md` | qual pasta é o método proposto e quais são baselines |
| `Code/Model/LBCF/README.md` | `unzip LBCF_RCT.zip` → seguir instruções em `./LBCF_RCT` → `LBCF_budget_allocation-RCT.py` |
| `Code/Model/LBCF/LBCF_RCT_extracted/UDCF_RCT/README.md` | procedimento de build (ver abaixo) |
| `Data/README.md` | descrição das bases |
| `Code/Evaluation/README.md` | quais métricas foram usadas |

### Procedimento de build documentado

Do README dentro de `LBCF_RCT`:

```
In build folder, run:
rm -r *
cmake ..
make
./UDCF

main entrance for this c++ project:  /core/main.cpp
you can modify the input data as you like.
```

Três observações:

- O `rm -r *` **é necessário**: o zip vem com um diretório `build/` já compilado, cujo
  `CMakeCache.txt` aponta para a máquina dos autores (`/home/yuqingwei/...`). Sem limpar,
  o cmake recusa configurar.
- O README diz `./UDCF`, mas o `CMakeLists.txt` declara `add_executable(UDCF_RCT ...)`.
  **O README está desatualizado**; o binário gerado chama-se `UDCF_RCT`.
- As duas últimas linhas são relevantes para a defesa metodológica: os autores
  **designam o `main.cpp` como ponto de entrada** (não como parte do algoritmo) e
  **autorizam explicitamente trocar os dados de entrada**.

### Descrição da base RCT dos autores

De `Data/README.md`, textualmente:

> *"web-scale RCT data collected from a **video streaming platform**... records the users'
> **campaign engagement duration** (i.e., outcome) in **seven randomly enrolled incentive
> groups**, each offered bonuses at different levels... over **100 K** app visit instances"*

O desfecho é **duração de engajamento** — uma variável contínua de escala alta (média
5.180 na amostra distribuída). O experimento real tem mais de 100 mil instâncias; a
amostra pública tem 2.000, criptografadas, e o README diz **"NOT FOR REPRODUCE THE
RESULT"**.

---

## 2. O modelo UDCF: artigo × código

### Para que serve

O artigo resolve **BTS** (*budget-constrained treatment selection*): N usuários, K
tratamentos com custos distintos, orçamento total limitado; decidir qual tratamento dar a
cada pessoa. A solução tem duas etapas:

1. **UDCF** — estima o CATE θ_ij de cada usuário i para cada tratamento j
2. **DGB** — resolve a alocação sob restrição orçamentária

Este trabalho cobre **apenas a etapa 1**.

### O que o UDCF substitui

A abordagem ingênua (**MBCF**) treina K florestas causais binárias separadas. O artigo
aponta duas limitações:

- custo de treinar e manter K florestas;
- **cada floresta particiona o espaço de covariáveis de forma diferente**, então os K
  efeitos estimados para a mesma pessoa vêm de regiões distintas — o que o artigo
  argumenta contradizer a definição de CATE.

Daí as duas propriedades: **Unified** (uma floresta só, todos os tratamentos sob as mesmas
regras de corte) e **Discriminative** (discrimina heterogeneidade entre nós e dentro do nó).

### O critério de divisão em duas etapas

**Inter split (Eq. 3):** maximiza heterogeneidade *entre* nós.

```
Δ_inter = Σ_filhos (1/n_l) · Σ_j (Σ_{i∈φl} ρ_ij)²
```

onde ρ_ij é o pseudo-desfecho — a contribuição individual de i para a diferença entre o
efeito do filho e o do pai. Barato: ρ é calculado uma vez por nó.

**Intra split (Eq. 4):** maximiza heterogeneidade *dentro* do nó.

```
Δ_intra = Σ_filhos Σ_j (θ̂_j − θ̄)²
```

onde θ̂_j é o efeito do tratamento j naquele filho e θ̄ a média dos K efeitos. Mede **quão
diferentes os K tratamentos são entre si**. Caro: exige reestimar θ para cada candidato.

**Combinação:** pegam os top *m* candidatos pelo Inter split (filtro barato) e escolhem
entre eles pelo Intra split (decisão cara). No código, *m* = 5% dos candidatos.

### Correspondência artigo ↔ código

| artigo | arquivo/linha | confere |
|---|---|---|
| Eq. 3 (Inter) | `UDCFSplittingRule.cpp:413-414` | sim |
| Eq. 4 (Intra) | `UDCFSplittingRule.cpp:247-261` | sim |
| top *m* pelo Inter, depois Intra | `UDCFSplittingRule.cpp:176` → `N = max(int(num_decrease*0.05),1)` | sim |
| Algoritmo 1 (cálculo de ρ) | `UDCFRelabelingStrategy::relabel` | sim |

**A implementação é fiel ao artigo.**

### O termo que está no código e não está no artigo

O código calcula:

```cpp
double decrease = sum_left.square().sum() / weight_sum_left +
                  (sum_node - sum_left).square().sum() / weight_sum_right;
double penalty_edge = imbalance_penalty * (1.0 / n_left + 1.0 / n_right);
decrease -= penalty_edge;
if (decrease > 0) { /* candidato aceito */ }
```

**A Eq. 3 não tem termo de penalização.** O `imbalance_penalty` vem da infraestrutura
herdada do GRF; o artigo apenas diz *"Termination rule for tree split is just the same as
BCF"*. Ver seção 6 para por que isso importa.

### Hiperparâmetros default

De `ForestTestUtilities::default_options(true, 1)`:

| parâmetro | valor |
|---|---|
| `num_trees` | 300 |
| `sample_fraction` | 0.5 |
| `mtry` | 3 |
| `min_node_size` | 50 |
| `honesty` | true |
| `honesty_fraction` | 0.5 |
| `honesty_prune_leaves` | true |
| `alpha` | 0.05 |
| `imbalance_penalty` | 0.01 |
| `num_threads` | 40 |
| `seed` | 42 |

**Nota sobre reprodutibilidade:** cada árvore recebe um `RandomSampler` próprio semeado a
partir de `seed + start`, onde `start` vem do fatiamento das árvores entre threads
(`ForestTrainer::train_batch`). Esse fatiamento depende de `num_threads`, que
`ForestOptions::validate_num_threads` só substitui pelo número de núcleos **quando vale 0**.
Como vale 40, o fatiamento é idêntico em qualquer máquina — os resultados independem do
hardware. Verificado empiricamente (seção 6).

---

## 3. Formato de entrada exigido pelo código

`load_data` (`commons/utility.cpp`) lê **texto puro, separado por espaços, sem cabeçalho e
sem índice**. Conta linhas para obter N e tokens da primeira linha para obter o número de
colunas.

Os papéis das colunas são declarados por índice:

```cpp
data.set_outcome_index(11);
data.set_treatment_index({12, 13});
```

**Ponto crítico** (`commons/Data.cpp:47-59`): esses métodos inserem as colunas em
`disallowed_split_variables`. É **só isso** que impede o modelo de dividir sobre o desfecho
ou o tratamento. Consequência: **toda coluna não declarada vira automaticamente candidata a
divisão**. Não existe declaração de covariável — covariável é o que sobra. Esquecer de
remover uma coluna pós-tratamento a transforma em preditor **sem nenhum aviso**.

**Tratamento multi-braço:** K colunas binárias, controle representado por **todas iguais a
zero**. Confirmado na base dos autores (7 colunas, 276 linhas de controle) e no script
`data_preprocessing.py` deles, cujas funções `getT1/getT2/getT3` atribuem `A==0` a todas as
colunas zeradas.

**Layout convencionado pelos autores** (de `data_preprocessing.py`):

```python
df[['H1','H2','H3','H4','Value','T1','T2','T3']].to_csv(
    "./data/train_data_...csv", header=None, index=False, sep=' ')
```

Ou seja: **covariáveis → desfecho → colunas de tratamento**.

### Por que o grupo de controle é obrigatório

O relabeling inverte W'W após centrar as colunas de tratamento. Se **toda** linha tivesse
exatamente um braço ativo, as K colunas somariam 1 sempre e, ao centrar, ficariam
linearmente dependentes.

Teste feito removendo o controle da base dos autores:

| | determinante | posto |
|---|---|---|
| com controle | 7,5 × 10¹⁵ | 7/7 |
| sem controle | 9,86 | **6/7** |

A deficiência de posto é real. **Mas o teste do código não a detecta**: a verificação é
`equal_doubles(WW_bar.determinant(), 0.0, 1.0e-10)`, e 9,86 é muito maior que 1e-10 — o
determinante "deveria" ser zero, mas o erro de arredondamento em valores dessa escala
produz 9,86. O modelo **inverte a matriz singular assim mesmo** e segue com resultados sem
sentido, sem emitir erro.

É o mesmo padrão do `imbalance_penalty`: **limiar absoluto onde deveria haver critério
relativo à escala**. O comentário dos próprios autores nessa linha diz *"This condition
number check works fine in practice - there may be more robust ways."*

### Layout usado para a Hillstrom

```
col  0-10  covariáveis (11): recency, history, mens, womens, newbie,
                             zip_Rural, zip_Surburban, zip_Urban,
                             channel_Multichannel, channel_Phone, channel_Web
col    11  conversion          <- outcome_index
col    12  T_mens              <- treatment_index[0]
col    13  T_womens            <- treatment_index[1]
```

Excluídas deliberadamente: `visit` e `spend` (pós-tratamento — seriam vazamento) e
`history_segment` (é o `history` discretizado — informação duplicada).

Grupos: controle 21.306 (taxa 0,573%), Mens E-Mail 21.307 (1,253%), Womens E-Mail 21.387
(0,884%).

---

## 4. As baselines do artigo

Todas produzem **CATEs num passo separado**, antes de qualquer otimização de orçamento. Em
todos os casos o `budget_allocation` é um script a jusante que apenas consome as predições —
pode ser ignorado sem prejuízo.

### 4.1 Chi, ED e CTS (`Code/Model/Chi_ED_CTS/`)

**O que são:** três florestas de uplift multi-tratamento, diferindo apenas no critério de
divisão — Chi-quadrado, Distância Euclidiana e *Contextual Treatment Selection*.

**Origem:** importadas do pacote **CausalML** da Uber (<https://github.com/uber/causalml>).
Não são reimplementações dos autores.

**Como rodar:** `python Chi_ED_CTS_train_and_predict-RCT.py` — só isso produz os CATEs.

**Configuração usada pelos autores:**

```python
UpliftRandomForestClassifier(
    n_estimators=300,           # = num_trees do UDCF
    evaluationFunction="Chi",   # ou "ED" ou "CTS"
    max_depth=5,                # UDCF não limita profundidade
    min_samples_leaf=100,
    min_samples_treatment=50,   # análogo ao min_node_size por braço do UDCF
    n_reg=100,
    control_name='0',
    n_jobs=1,
    normalization=True)
```

**Saída:**

```python
pred = model.predict(test[features].values)
pd.DataFrame(pred, columns=model.classes_).to_csv('RCT_Chiresult')
```

Uma linha por indivíduo, **uma coluna por braço de tratamento** — mesma forma da saída do
UDCF, diretamente comparável.

**Ponto de atenção:** `min_samples_treatment=50` é a mesma exigência de mínimo **por braço**
que degenera o UDCF na amostra dos autores (seção 6). Pode restringir essas baselines
também, dependendo da base.

**Esforço para adaptar à Hillstrom:** baixo. `pip install causalml`, trocar o caminho dos
dados e a lista de features.

### 4.2 CF_DT / MBCF (`Code/Model/CF_DT/`)

**O que é:** *Multiple Binary Causal Forests* — treina **K florestas causais binárias
independentes**, uma por braço contra o controle. É precisamente a abordagem que o UDCF foi
criado para substituir.

**Por que é a comparação mais importante:** toda a justificativa do artigo para o UDCF é
consertar as limitações do MBCF. Comparar os dois na Hillstrom **testa a afirmação central
do artigo** em dados novos.

**Como rodar:** `unzip CF_DT_RCT.zip` → seguir instruções em `./MBCF_RCT` (mesmo
procedimento de build do UDCF) → `python data_merging.py`.

**Saída:** um arquivo por braço (`MBCF_uplift_RCT1.csv` … `RCT7.csv`), unidos depois pelo
`data_merging.py` em colunas `predict_1 … predict_K`.

**Configuração usada pelos autores** (do `main.cpp` em `CF_DT_RCT.zip`):

```cpp
data.set_outcome_index(14);
data.set_treatment_index(15);
data.set_instrument_index(15);   // instrumento = tratamento
double reduced_form_weight = 0.0;
bool stabilize_splits = false;   // -> RegressionSplittingRuleFactory
ForestTrainer trainer = instrumental_trainer(reduced_form_weight, stabilize_splits);
ForestOptions options = ForestTestUtilities::default_options(true, 1);
```

Usar o tratamento como seu próprio instrumento é válido por se tratar de experimento
aleatorizado, e é o que torna isso uma floresta causal binária padrão.

**Observação relevante:** o código-fonte em `src/` do `CF_DT_RCT.zip` é **idêntico** ao do
`LBCF_RCT.zip` (verificado com `diff -rq`). Os dois modelos compartilham a mesma base GRF;
só mudam o `main.cpp` e o nome do alvo no CMake. Isso explica por que os zips do UDCF
aparecem dentro da pasta `CF_DT` (seção 1) — e significa que uma única compilação serve
aos dois modelos.

**Esforço:** médio. É C++, mas o toolchain necessário é o mesmo já usado para o UDCF.

### 4.3 CT_ST (`Code/Model/CT_ST/`)

**O que é:** causal tree com otimização estocástica, de Tu et al. (2021).

**Origem:** código em **R**, de <https://github.com/tuye0305/prophet>, mais Python para
processamento.

**Como rodar:** `python dataPrepRCT.py` → `Rscript main.R` → `python predRCT.py`. Exige
configurar `homePath` no `main.R` e montar um ambiente R.

**Esforço:** alto — é o único que exige uma linguagem a mais.

### 4.4 Resumo comparativo

| baseline | linguagem | multi-tratamento | esforço | interesse científico |
|---|---|---|---|---|
| Chi / ED / CTS | Python (CausalML) | sim, nativo | baixo | médio — 3 modelos de uma vez |
| CF_DT (MBCF) | C++ | sim, via K florestas | médio | **alto** — testa a tese do artigo |
| CT_ST | R + Python | sim | alto | médio |

---

## 5. Como os autores avaliam os modelos

De `Code/Evaluation/README.md`:

| contexto | métrica | script |
|---|---|---|
| dados sintéticos (Seção 5.1) | medida de **ITE** | `Simulation_analysis.py` |
| dados reais (Seção 5.2) | **PMG** — *percentage mean gain* | `Offline_test.py` |

No sintético eles conhecem o efeito verdadeiro (geraram os dados), então avaliam o CATE
diretamente. Em dados reais isso é impossível.

### Como o PMG funciona

Decodificando `Offline_test.py`:

1. a etapa DGB atribui a cada usuário um tratamento recomendado (`final_coin`);
2. `treatment` é o tratamento que a pessoa **de fato recebeu** no experimento;
3. `if_same` marca quem recebeu exatamente o que a política recomendaria;
4. o ganho médio da política é estimado **apenas sobre esse subconjunto coincidente**;
5. `PMG = (ganho_médio − base) / base`.

É avaliação de política *offline* por coincidência com a aleatorização.

### Implicação para este trabalho

**O PMG avalia a política de alocação, não as estimativas de CATE.** Depende de
`final_coin`, que só existe após rodar o DGB — a etapa 2, fora do escopo deste trabalho.

Consequência: **a combinação "apenas UDCF + dados reais" não tem métrica definida no
artigo.** Os autores avaliam CATE só onde há gabarito (sintético) e, em dados reais,
avaliam o sistema completo.

Isso **justifica** adotar instrumental de avaliação externo ao artigo (calibração,
ordenação, curvas de uplift) — e essa justificativa precisa estar explícita no texto.

---

## 6. Achados na aplicação à Hillstrom

### 6.1 Com os hiperparâmetros default, não há nenhuma divisão

```
DIAG_NODES     total_nodes=300  internal_nodes=0  stump_trees=300
DIAG_SPLIT_TOTAL 0
DIAG_SPLIT_FREQ  0:0 1:0 2:0 3:0 4:0 5:0 6:0 7:0 8:0 9:0 10:0
```

As 300 árvores são tocos; nenhuma das 11 covariáveis é usada. O CATE é constante.

A dispersão residual observada nas predições (desvio 0,000091 sobre média 0,006716, isto é
1,35%) é **artefato da predição out-of-bag** — cada observação é predita por um subconjunto
diferente de árvores — e não heterogeneidade modelada.

**Duas confirmações independentes**, que não dependem da contagem de divisões:

- o CATE médio (0,006716) coincide com a diferença simples de médias (0,006805);
- o CATE médio por subgrupo é idêntico até a 6ª casa decimal em todas as covariáveis
  categóricas (`zip_code`: 0,006715 / 0,006716 / 0,006716; idem `channel` e `newbie`).

**Corolário:** sob esta configuração a codificação das categóricas é irrelevante. One-hot e
ordinal produzem resultados idênticos (diferença absoluta máxima = 0), porque sem divisões
nenhuma covariável influencia a estimativa.

### 6.2 O mesmo ocorre na base dos autores

Rodando o `main.cpp` **original, sem uma linha alterada**, na base RCT dos próprios autores:

```
2.000 predições  →  1 linha única
-543.618, 212.479, -314.761, 257.846, 669.643, 148.965, 389.158
```

O fenômeno **não é específico da Hillstrom**.

### 6.3 As causas são diferentes em cada base

Experimento fatorial 2×2 na Hillstrom, com tudo o mais constante (mesma semente, mesmos
dados, mesmo binário):

| | `imbalance_penalty = 0,01` | `imbalance_penalty = 0` |
|---|---|---|
| `min_node_size = 50` | **0 splits** · 300 tocos | 11.407 splits · 0 tocos |
| `min_node_size = 5` | **0 splits** · 300 tocos | 28.988 splits · 0 tocos |

Na Hillstrom, `min_node_size` é irrelevante; o que trava é o `imbalance_penalty`.

Na base dos autores, o oposto: com `imbalance_penalty` mantido em 0,01, reduzir
`min_node_size` de 50 para 5 destrava **1.398 splits**.

**Por quê:** `min_node_size` é aplicado **por braço de tratamento e dos dois lados do
corte**, exigindo ≥ 2× o valor por braço no nó pai. Com 2.000 linhas, 7 braços,
`sample_fraction=0.5` e `honesty_fraction=0.5`, cada árvore cresce com ~500 observações,
~60 por braço, contra as 100 exigidas — nenhum corte é geometricamente possível.

### 6.4 O mecanismo do `imbalance_penalty`

Com pesos = 1 (`Data::get_weight` retorna 1,0 quando não há coluna de pesos) e soma dos
pseudo-desfechos igual a zero sobre o nó (condição de primeira ordem do OLS no relabeling,
verificada numericamente: ~1e-17), a condição de aceitação fatora exatamente:

```
decrease > 0   ⟺   ‖s_esq‖² > imbalance_penalty
```

O limiar é **absoluto**, enquanto o ganho escala com a variância do desfecho. O critério
não é invariante à escala da resposta.

| base | n | braços | desvio do desfecho | max ‖s_esq‖² | limiar | há split? |
|---|---|---|---|---|---|---|
| Sintética | 80.000 | 3 | 540,5 | — | 0,01 | **sim** |
| RCT (autores) | 2.000 | 7 | 7.128,3 | 3,80 × 10⁶ | 0,01 | bloqueada antes, por `min_node_size` |
| **Hillstrom** | 64.000 | 2 | **0,095** | **1,93 × 10⁻⁶** | 0,01 | **não** |

Cerca de **doze ordens de grandeza** separam a base dos autores da Hillstrom na mesma
quantidade. Seria necessário multiplicar o desfecho por ~72 para haver qualquer divisão.
Com `spend` como desfecho (desvio 15,0) o critério passa.

**Propriedade contraintuitiva:** como ρ ~ 1/n, a quantidade ‖s_esq‖² *encolhe* conforme n
cresce. Sob limiar absoluto, **mais dados tornam o split mais difícil**. Verificado:
n=2.000 → 9,4e-5; n=16.000 → 5,6e-6; n=64.000 → 1,9e-6.

### 6.5 Atribuição correta da responsabilidade

O teste `decrease > 0` é **comportamento original do GRF**, não modificação dos autores do
UDCF — ver `InstrumentalSplittingRule.cpp`, que inicializa `best_decrease = 0.0` e faz
`if (best_decrease <= 0.0) return true;`.

O que torna o critério restritivo é a **combinação** desse teste com um valor não nulo de
`imbalance_penalty` nos defaults, aplicada a um desfecho de escala muito inferior.

E, como visto na seção 2, **a Eq. 3 do artigo não contém termo de penalização**. Portanto
`imbalance_penalty = 0` não afrouxa o modelo: faz o código computar o critério tal como
publicado.

### 6.6 Avaliação das estimativas obtidas com `imbalance_penalty = 0`

| configuração | BLP | p | Q5 − Q1 | p | placebo |
|---|---|---|---|---|---|
| Mens E-Mail | 0,015 | 0,967 | 0,0026 | 0,382 | dentro do nulo |
| Womens E-Mail | 0,401 | 0,196 | 0,0041 | 0,118 | dentro do nulo |
| Mens, config. original (degenerada) | **−94,3** | <0,001 | −0,0243 | <0,001 | fora do nulo |

**A heterogeneidade não é estatisticamente detectável.** Teste conjunto das 11 interações
sobre os dados brutos: p = 0,19. Com taxa de conversão de 0,57% no controle, a menor
diferença detectável entre subgrupos é ~0,0029, comparável ao próprio efeito médio de
0,0068.

**Redação obrigatória:** "não detectável", nunca "inexistente". Validação em dados
sintéticos mostrou que um sinal real porém atenuado **não seria detectado** neste tamanho
de amostra.

### 6.7 Armadilha metodológica: o modelo degenerado "passa" no teste de calibração

A terceira linha da tabela acima merece registro. A floresta **sem nenhuma divisão** —
que por construção não pode ter aprendido estrutura — produz BLP = −94 com p < 0,0001.

É artefato do `predict_oob`: a estimativa de cada indivíduo é a média sobre as árvores que
**o excluíram** da amostra. Um tratado que converteu eleva o efeito global, logo as árvores
sem ele estimam menos; um controle que converteu produz o efeito oposto.

| grupo | n | CATE médio |
|---|---|---|
| tratado, converteu | 267 | 0,00667498 |
| tratado, não converteu | 21.040 | 0,00671691 |
| controle, converteu | 122 | 0,00676101 |
| controle, não converteu | 21.184 | 0,00671525 |

Contraste entre tratados: −0,0000419 (p < 10⁻¹²). Entre controles: +0,0000458 (p < 10⁻⁸).
**Sinais opostos** — a assinatura prevista pelo mecanismo. Na floresta com divisões o padrão
é diferente (ambos positivos), compatível com efeito de covariável e não com vazamento.

**Conclusão metodológica:** o teste de calibração pode produzir resultado altamente
significativo sobre um modelo degenerado. Sua interpretação deve sempre vir acompanhada da
verificação de que o modelo efetivamente realizou divisões.

### 6.8 Onde está a heterogeneidade detectável na Hillstrom

Testes de interação diretamente sobre os dados brutos, sem usar modelo:

| covariável | separação entre braços (θ_mens − θ_womens) | p |
|---|---|---|
| **`mens`** | 0,0006 (mens=0) → **0,0062** (mens=1) | **0,023** |
| `history` | 0,0004 (Q1) → 0,0063 (Q4) | 0,107 |
| `womens` | 0,0054 → 0,0023 | 0,212 |
| `newbie` | 0,0053 → 0,0021 | 0,202 |
| `recency` | 0,0036 → 0,0025 | 0,775 |

A única covariável com sinal discriminativo significativo é `mens` — e ela **não aparece
entre as seis variáveis mais usadas** pela floresta, que gastou 30% dos splits em `history`
e 20% em `recency`.

Explicação provável: viés por alta cardinalidade. `history` tem milhares de valores únicos,
logo milhares de pontos de corte candidatos; `mens` é binária e oferece um só.

**Ressalva:** são cinco testes, e `mens` a p=0,023 não sobreviveria a correção de
Bonferroni. Evidência sugestiva, não conclusiva.

Separadamente, o efeito conhecido da base aparece com força: o e-mail feminino tem ATE
0,0051 em quem tem histórico feminino contra 0,0006 em quem não tem (p = 0,0055; no
desfecho `visit`, z = 9,7). Mas esse é um efeito **de um braço isolado**, não de separação
entre braços — que é o que o UDCF busca.

### 6.9 Comparação com baselines

Seis configurações, avaliadas sobre **as mesmas 64.000 linhas**, com predições
fora-da-amostra em todos os casos: `predict_oob` para os modelos em C++, validação
cruzada de 5 dobras estratificada para os do CausalML. As baselines usam os
hiperparâmetros exatos que os autores do LBCF configuraram para elas.

Protocolo: `predict_oob` para os modelos em C++ (UDCF, ablação, MBCF), validação cruzada
de 5 dobras estratificada para os do CausalML. Cada braço é avaliado sobre as linhas de
controle mais as daquele braço.

**Mens E-Mail** (ATE ingênuo 0,006805; n = 42.613)

| modelo | splits | desvio | BLP | p | Q5−Q1 | p | placebo p |
|---|---|---|---|---|---|---|---|
| UDCF default | **0** | 0,000091 | −94,3 | 0,000 | −0,0243 | 0,000 | 0,000 |
| UDCF `ip=0` | 11.407 | 0,002739 | 0,015 | 0,967 | 0,00263 | 0,382 | 0,973 |
| Ablação `ip=0` | 28.375 | 0,003998 | 0,024 | 0,922 | 0,00081 | 0,798 | 0,910 |
| Chi | — | 0,002801 | 0,446 | 0,200 | 0,00279 | 0,358 | 0,217 |
| ED | — | 0,003998 | 0,314 | 0,234 | 0,00212 | 0,488 | 0,257 |
| CTS | — | 0,002583 | 0,260 | 0,468 | 0,00394 | 0,200 | 0,423 |
| MBCF default | 19.239 | 0,003084 | 0,068 | 0,830 | 0,00244 | 0,438 | 0,820 |
| MBCF `ip=0` | 23.615 | 0,003531 | −0,046 | 0,866 | −0,00032 | 0,918 | 0,873 |

**Womens E-Mail** (ATE ingênuo 0,003111; n = 42.693)

| modelo | splits | desvio | BLP | p | Q5−Q1 | p | placebo p |
|---|---|---|---|---|---|---|---|
| UDCF default | **0** | 0,000083 | −90,4 | 0,000 | −0,0219 | 0,000 | 0,000 |
| UDCF `ip=0` | 11.407 | 0,002857 | 0,401 | 0,196 | 0,00406 | 0,118 | 0,213 |
| Ablação `ip=0` | 28.375 | 0,003767 | −0,038 | 0,882 | 0,00107 | 0,702 | 0,850 |
| **Chi** | — | 0,003392 | **0,523** | **0,047** | **0,00708** | **0,011** | 0,053 |
| ED | — | 0,003718 | 0,213 | 0,424 | 0,00379 | 0,184 | 0,480 |
| CTS | — | 0,002345 | −0,118 | 0,766 | −0,00007 | 0,980 | 0,737 |
| MBCF default | 17.105 | 0,003039 | 0,305 | 0,319 | 0,00207 | 0,447 | 0,310 |
| MBCF `ip=0` | 21.178 | 0,003507 | 0,038 | 0,888 | 0,00237 | 0,398 | 0,870 |

**Dois fatos centrais.**

Primeiro: **o UDCF é o único modelo que degenera com os hiperparâmetros default dos
autores.** MBCF, Chi, ED e CTS produzem heterogeneidade normalmente com as configurações
que os próprios autores do LBCF definiram para eles.

Segundo: **nenhum modelo demonstra heterogeneidade estatisticamente detectável.** São 14
testes legítimos (7 configurações não-degeneradas × 2 braços), e apenas um atinge
p < 0,05 — o Chi no braço feminino, que é aproximadamente o que o acaso entrega em 14
testes e não sobrevive a Bonferroni (limiar 0,0036). Seu placebo dá p = 0,053, na
fronteira. Redação correta: *indício de heterogeneidade no braço feminino, que não
sobrevive à correção para múltiplas comparações*.

Coerente com o teste conjunto das 11 interações nos dados brutos (p = 0,19) e com o limite
de poder da amostra.

**Padrão não estabelecido, registrado por transparência:** em 3 das 4 comparações pareadas
em que uma versão corta mais que a outra, a que corta **menos** apresenta BLP maior
(UDCF 11.407 → 0,401 vs ablação 28.375 → −0,038; MBCF 19.239 → 0,305 vs MBCF 23.615 →
0,038). O braço masculino contradiz num dos pares, e todas as diferenças são
não-significativas. Fraco demais para afirmar; anotado apenas para não omitir.

### 6.10 Por que o UDCF degenera e os outros não

A explicação tem **dois ingredientes**, e apenas o segundo é específico do UDCF.

**Ingrediente 1 — o limiar é absoluto.** Vem do GRF e vale para toda a família: o
candidato só é aceito se `decrease > 0`, e `decrease` subtrai
`imbalance_penalty × (1/n_esq + 1/n_dir)`. Presente no `UDCFSplittingRule`, no
`MultiRegressionSplittingRule` e no `RegressionSplittingRule` igualmente.

**Ingrediente 2 — a normalização do relabeling.** Aqui os dois divergem:

```cpp
// UDCFRelabelingStrategy.cpp:70     rho_weight = W_centrado * A^-1,  A = W'W ~ O(n)
responses_by_sample(sample, j) = rho_weight(i, treatment) * residual(i, outcome);

// InstrumentalRelabelingStrategy.cpp:94     sem normalizacao alguma
responses_by_sample(sample, 0) = (instrumento - media) * residual;
```

O relabeling multi-braço **divide por `(W'W)⁻¹`, que escala como 1/n**. O instrumental
não divide por nada.

Verificação numérica na Hillstrom, calculando os dois relabelings sobre nós do mesmo
tamanho:

| n do nó | UDCF \|ρ\| médio | Instrumental \|ρ\| médio | razão |
|---|---|---|---|
| 40.000 | 8,14 × 10⁻⁷ | 9,00 × 10⁻³ | 11.059× |
| 16.000 | 1,98 × 10⁻⁶ | 9,20 × 10⁻³ | 4.654× |
| 4.000 | 6,69 × 10⁻⁶ | 9,39 × 10⁻³ | 1.405× |
| 1.000 | 6,20 × 10⁻⁵ | 9,94 × 10⁻³ | 160× |

O instrumental fica **constante em ~0,009 independente de n**; o do UDCF **encolhe**
conforme o nó cresce. E a consequência, num nó de 16.000 amostras:

```
UDCF         max‖s_esq‖² = 4,83 × 10⁻⁶   →  reprovado no limiar de 0,01
Instrumental max‖s_esq‖² = 3,80 × 10¹    →  aprovado com folga de 3.800×
```

**Conclusão.** Não é que o UDCF seja inferior, nem que os defaults do GRF sejam ruins em
geral. É que o critério multi-braço tem uma **dependência do tamanho do nó** que o critério
binário não tem, e essa dependência interage mal com um limiar absoluto herdado. Em
amostra pequena com desfecho de escala grande — os 2.000 registros e desfecho na casa dos
milhares dos autores — o fator 1/n não morde. Em amostra grande com desfecho binário raro,
morde duas vezes.

Isso também explica a propriedade contraintuitiva registrada em 6.4: **mais dados tornam o
split mais difícil**. O 1/n vem daqui.

**Registro de correção.** Uma versão anterior destas notas atribuía a diferença ao fato de
os critérios do CausalML serem medidas de divergência sem limiar absoluto. Essa explicação
estava incompleta: não dava conta do MBCF, que **tem** o limiar absoluto e ainda assim
produz 19.239 divisões. A explicação acima, verificada numericamente, substitui aquela.

### 6.11 Ablação: o Intra split faz diferença?

O parâmetro `stabilize_splits` de `udcf_trainer` seleciona a regra de divisão:

```cpp
udcf_trainer(K, 1, true)   // UDCFSplittingRule           — Inter + Intra
udcf_trainer(K, 1, false)  // MultiRegressionSplittingRule — apenas Inter
```

Com `false`, o relabeling é o mesmo e o critério Inter é o mesmo; apenas o segundo estágio
(Intra split, a contribuição do artigo) desaparece. É uma ablação exata, disponível no
próprio código dos autores.

| | UDCF (Inter+Intra) | Ablação (só Inter) |
|---|---|---|
| `ip = 0,01` | 0 splits · 300 tocos | **0 splits · 300 tocos** |
| `ip = 0` | 11.407 splits | **28.375 splits** |

Com os defaults, **os dois degeneram igualmente** — o Intra split não tem
responsabilidade na degeneração. Com `ip = 0`, a ablação corta 2,5× mais, e a distribuição
por variável é quase idêntica em proporção (`history` 30,0% vs 30,4%; `recency` 20,0% vs
20,5%): o critério discriminativo **não redireciona a floresta para outras covariáveis**
nesta base, apenas corta menos.

Nas métricas (tabelas 6.9), nem UDCF nem ablação atingem significância. O UDCF aparenta
ordenar melhor no braço feminino (BLP 0,401 vs −0,038), mas com p = 0,196 e p = 0,882 —
**as duas são não-significativas, logo a diferença entre elas também não está
estabelecida**.

**Resposta à pergunta "o Intra split ajuda na Hillstrom?": não de forma detectável.**

### 6.12 Contagem de braços, e por que isso limita o alcance do teste

**Convenção: K = número de grupos experimentais − 1.** O controle é representado pela
ausência de todas as colunas de tratamento, nunca por uma coluna própria.

| | grupos | K |
|---|---|---|
| Autores | 8 (`exp_group` 0–7; contagens 276, 231, 255, 259, 237, 251, 227, 264) | 7 |
| Hillstrom | 3 (No E-Mail, Mens, Womens) | **2** |

O `exp_group = 0` dos autores tem 276 linhas, que são exatamente as 276 com as sete
colunas de tratamento zeradas. Nota de desenho: esse grupo **não** é ausência de
tratamento — pelo `coin_map` do `LBCF_budget_allocation-RCT.py` ele recebe bônus de 0,1, o
menor nível. Na Hillstrom, `No E-Mail` é controle verdadeiro.

**Implicação para o resultado da ablação (6.11).** O critério Intra split mede a variância
entre os K efeitos dentro do nó. Com K = 7 há muitas formas de os tratamentos se
diferenciarem; com **K = 2 a expressão colapsa para (θ₁ − θ₂)²/2** — um único grau de
liberdade. Uma explicação plausível para o Intra split não fazer diferença detectável é
que, com dois braços, ele tem pouco o que discriminar. A contribuição do artigo foi
projetada para o cenário de muitos tratamentos, que não é o da Hillstrom.

### 6.13 Segundo desfecho: `visit`

A base Hillstrom traz três desfechos pós-tratamento: `visit`, `conversion` e `spend`. O
trabalho usa `conversion`. Rodou-se **todo o pipeline também com `visit`**, não para trocar
o desfecho, mas para obter um contraste de poder estatístico dentro da mesma base, com as
mesmas covariáveis e o mesmo experimento.

| | `conversion` | `visit` |
|---|---|---|
| taxa geral | 0,903% | **14,678%** |
| controle | 0,573% | 10,617% |
| Mens E-Mail | 1,253% | 18,276% |
| Womens E-Mail | 0,884% | 15,140% |
| ATE mens | +0,681 pp | **+7,659 pp** |
| ATE womens | +0,311 pp | **+4,523 pp** |

Dezesseis vezes mais eventos, efeito cerca de onze vezes maior.

### 6.14 O UDCF degenera com `visit` também — e isso isola a causa

| desfecho | `imbalance_penalty` | splits | tocos |
|---|---|---|---|
| `conversion` | 0,01 | **0** | 300/300 |
| **`visit`** | 0,01 | **0** | **300/300** |
| `conversion` | 0 | 11.407 | 0 |
| `visit` | 0 | 11.681 | 0 |

A ablação (sem Intra split) degenera igualmente com `ip = 0,01` nos dois desfechos, e faz
38.618 divisões com `ip = 0` em `visit`.

**Consequência.** Uma hipótese anterior — de que a raridade do evento contribuía para a
degeneração — **fica descartada**. Trocar o desfecho multiplicou a escala do resíduo por
cerca de 4 e o ganho por cerca de 14, mas a distância até o limiar era de **quatro ordens
de grandeza**. Não chega perto.

Sobra o mecanismo estrutural puro descrito em 6.10: normalização por `(W'W)⁻¹ ~ 1/n`
contra um limiar absoluto. **A degeneração não depende das particularidades da Hillstrom** —
ocorrerá em qualquer base cujo `n` seja grande o bastante. A base dos autores escapa por
ter apenas 2.000 linhas e desfecho na casa dos milhares.

### 6.15 Avaliação orientada à decisão: valor da política

As métricas de ordenação (BLP, GATES, Qini, RATE) avaliam **um braço de cada vez**: *este
modelo ordena bem as pessoas para o e-mail masculino contra não enviar nada?* Nenhuma
delas toca a segunda camada da decisão — *qual* e-mail —, que é exatamente o que o Intra
split do UDCF otimiza.

Adotou-se, portanto, uma avaliação de **valor de política**. A política de um modelo
atribui a cada pessoa uma ação em {nada, mens, womens} pelo maior CATE estimado. O valor é
estimado nas pessoas cujo braço **sorteado** coincidiu com o recomendado — estimador de
Hájek, válido porque a atribuição é aleatorizada e independente das covariáveis. É a mesma
lógica do PMG dos autores, sem exigir custo nem orçamento (a Hillstrom não traz custo por
tratamento, o que torna o DGB inaplicável por falta de dado, não apenas por escopo).

Implementação: `causalml.optimize` (`get_uplift_best`, `get_actual_value`), nativa para
múltiplos tratamentos.

**Políticas de referência:**

| referência | o que estabelece |
|---|---|
| não abordar ninguém | o piso |
| **`mens` para todos** | **a barra honesta** — o melhor sem personalização |
| `womens` para todos | o outro braço fixo |
| braço sorteado para todos | personalizar sem informação |
| mesmas pessoas, braço sorteado | *(descartada — ver abaixo)* |

**Ressalva registrada.** A quinta referência, inicialmente proposta para isolar a camada 2,
mostrou-se **contaminada**: o UDCF degenerado a superava em +0,184 pp, apesar de recomendar
`mens` para 100% das pessoas. O ganho era apenas o efeito marginal de `mens` > `womens`,
não personalização. A referência honesta para a camada 2 é **`mens` para todos**.

**Resultados (desfecho `visit`; barra = 18,276%; erro-padrão 0,263 pp):**

| modelo | % que manda `mens` | valor | vs. barra |
|---|---|---|---|
| CTS | 82,4% | 18,381% | +0,105 |
| UDCF `ip=0` | 82,9% | 18,334% | +0,058 |
| UDCF default | **100,0%** | 18,276% | +0,000 |
| MBCF | 79,8% | 18,234% | −0,042 |
| Ablação `ip=0` | 78,1% | 18,210% | −0,065 |
| ED | 72,4% | 18,210% | −0,065 |
| Chi | 79,0% | 18,201% | −0,075 |
| S-learner | 81,5% | 18,159% | −0,117 |
| T-learner | 77,2% | 18,022% | −0,254 |

A maior vantagem é de **0,40 erro-padrão**. Nenhum modelo supera mandar `mens` para todos.
Com `conversion` o quadro é o mesmo (melhor: UDCF `ip=0`, +0,029 pp contra 0,073 pp de
erro-padrão).

**Curva de cobertura** (tratar os top q% pelo maior CATE): sobe monotonicamente em todos os
modelos e nos dois desfechos. Não existe fração de cobertura que supere tratar toda a base.

**O UDCF degenerado recomenda `mens` para 100% das pessoas** — com CATE constante o argmax
é sempre o mesmo braço. Sua "política personalizada" é literalmente a política cega, e
pontua exatamente a barra. Ilustração de que política sem heterogeneidade colapsa para a
ação marginal.

### 6.16 Com `visit`, a heterogeneidade existe e todos os modelos a encontram

Qini com nulo por permutação (`normalize=False`, 400 permutações) e GATES por quintil:

**Braço `womens`** (ATE ingênuo +4,523 pp):

| modelo | Qini | p | GATES Q1→Q5 (pp) | Q5−Q1 | p |
|---|---|---|---|---|---|
| Chi | 147,8 | 0,000 | 2,1 · 0,7 · 5,3 · 8,0 · 6,6 | +4,54 | 0,000 |
| S-learner | 145,0 | 0,000 | 2,2 · 0,5 · 4,8 · 8,1 · 7,0 | +4,80 | 0,000 |
| Ablação `ip=0` | 144,0 | 0,000 | 2,3 · 1,6 · 3,6 · 8,7 · 6,5 | +4,20 | 0,000 |
| ED | 143,9 | 0,000 | 2,1 · 0,2 · 5,4 · 8,7 · 6,2 | +4,05 | 0,000 |
| MBCF | 142,5 | 0,000 | 2,2 · 1,2 · 4,4 · 8,0 · 6,7 | +4,49 | 0,000 |
| T-learner | 141,4 | 0,000 | 2,4 · 0,6 · 4,8 · 7,5 · 7,3 | +4,86 | 0,000 |
| UDCF `ip=0` | 135,4 | 0,000 | 2,3 · 0,7 · 4,9 · 8,2 · 6,6 | +4,30 | 0,000 |
| CTS | 125,6 | 0,000 | 2,5 · 2,0 · 3,1 · 8,8 · 6,1 | +3,64 | 0,000 |
| **UDCF default** | **−160,8** | 1,000 | 8,2 · 6,1 · 4,5 · 4,0 · −0,1 | **−8,29** | **0,000** |

**Braço `mens`** (ATE ingênuo +7,659 pp): nenhum modelo atinge significância (o melhor é
CTS, p = 0,055 no Qini e 0,063 no GATES); os perfis de GATES são praticamente planos.

Ou seja: **a heterogeneidade da Hillstrom está no braço feminino**, e todos os oito modelos
funcionais a encontram, com desempenho muito próximo (Qini de 125,6 a 147,8). Isso é
coerente com o teste direto nos dados brutos (seção 6.8): o e-mail feminino tem efeito de
7,31% em quem tem histórico feminino contra 1,11% em quem não tem, com z = 9,7 no `visit`.

### 6.17 Detectar heterogeneidade não é o mesmo que poder explorá-la

Os dois resultados acima parecem contraditórios — heterogeneidade altamente significativa,
mas nenhum ganho de política. A aritmética resolve:

- o melhor quintil respondedor ao e-mail **feminino** recebe **~8,2 pp**;
- mandar o e-mail **masculino** para qualquer pessoa já entrega **+7,659 pp** em média.

O ganho de identificar corretamente quem responde ao feminino é de **~0,5 pp, e apenas
naquele quintil**. Diluído na base, aproximadamente **0,1 pp** — exatamente a ordem de
grandeza das diferenças observadas na tabela de política.

**A heterogeneidade é real, é detectável, e é inexplorável** — porque o braço onde ela
existe é dominado em média pelo outro braço. É um achado aplicado que merece destaque: em
um cenário multi-tratamento, a utilidade da personalização depende não só de existir
heterogeneidade, mas de ela ser grande o bastante para reverter a ordem marginal dos
tratamentos.

**Correção registrada.** A expectativa declarada antes de rodar era de que o desfecho
`visit` destravaria a camada 2 e permitiria distinguir os modelos por valor de política.
**Não destravou** — a resposta de aplicação é idêntica à de `conversion`. O que a troca
entregou foi outra coisa, e mais valiosa: (i) descartou a raridade do evento como causa da
degeneração do UDCF, e (ii) elevou o poder estatístico a ponto de "não há ganho em
personalizar" deixar de ser inconclusivo e passar a ser um resultado sustentado.

### 6.18 Figura

`curvas_qini.png` (raiz do repositório): curvas Qini em múltiplos pequenos, um painel por
modelo, para os dois braços, desfecho `conversion`. A linha tracejada é o ranking
aleatório; acima dela o modelo prioriza melhor que o acaso. O painel do UDCF default
mergulha abaixo da diagonal — até −42 respondentes incrementais no braço feminino —,
tornando visível o artefato descrito em 6.7.

*Nota de execução: o validador de paleta do procedimento de visualização não pôde ser
executado (Node.js ausente no ambiente). Usaram-se os valores documentados da paleta de
referência sem alteração, e o desenho em múltiplos pequenos dispensa codificação
categórica de cor.*

---

## 7. Estado de verificação

### Verificado

| afirmação | como |
|---|---|
| pasta e zip corretos | `Code/Model/README.md`, `LBCF/README.md`, MD5 |
| procedimento de build | README de `LBCF_RCT` |
| formato de entrada | `data_preprocessing.py` dos autores |
| algoritmo intocado | os 6 arquivos do UDCF nunca editados |
| degeneração na base dos autores | `main.cpp` original, MD5 conferido, executado |
| causa na Hillstrom | experimento fatorial 2×2 |
| independência de ambiente | Colab (g++ ~11/13) × WSL (g++ 15.2): 11.407 splits, distribuição idêntica variável por variável, primeira predição igual até o último dígito |

### Não verificado

**Não comparamos nossa saída com um resultado de referência publicado pelos autores.** O
teste previsto — rodar o código sintético sem alteração e comparar com os
`UDCF_uplift_*uw.csv` que vêm no repositório — foi interrompido por falta de memória antes
de gravar qualquer arquivo. Decidiu-se não retomá-lo por restrição de espaço no texto.

Consequência: as verificações existentes são de **consistência interna** (nossas execuções
concordam entre si e com o procedimento documentado), não de **reprodução de resultado
publicado**.

### Divergência deliberada em relação aos autores

O `main.cpp` original dos autores para dados RCT usa
`predictor.predict(forest, data, data2, false)` — treina numa base e prediz em outra
separada. Este trabalho usa **`predict_oob` sobre 100% dos dados**. É escolha
metodológica defensável (predição honesta por construção, aproveitando toda a amostra),
mas **é divergência e precisa constar na metodologia**.

---

## 8. Ambiente de execução

| item | versão |
|---|---|
| WSL2 + Ubuntu | 26.04.1 LTS |
| g++ | 15.2.0 |
| cmake | 4.2.3 |
| make | 4.4.1 |

**Ajuste necessário:** o `CMakeLists.txt` dos autores declara
`cmake_minimum_required(VERSION 2.0)`, e o CMake 4 removeu compatibilidade com versões
abaixo de 3.5. Contornado com o flag de linha de comando
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` — **nenhum arquivo dos autores foi alterado**. No
Colab o problema não aparece porque lá o CMake é 3.x.

**Avisos de compilação:** todos originários de `third_party/random/random.hpp`
(`__x` e `__difmuk` possivelmente não inicializadas na distribuição de Poisson). É
biblioteca de terceiros embutida, não código dos autores. A função afetada
(`sample_poisson`) é usada no sorteio de variáveis candidatas por nó. Se houvesse
comportamento indefinido real afetando os sorteios, os resultados no Colab e no WSL não
coincidiriam — coincidem exatamente, o que torna a hipótese improvável.
