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
