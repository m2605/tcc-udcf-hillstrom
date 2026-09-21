# UDCF aplicado à base Hillstrom 

Aplicação do modelo **UDCF** (*Unified Discriminative Causal Forest*) à base pública de
marketing Hillstrom, cobrindo a etapa de estimação de CATE.

## Sobre o modelo e o código do algoritmo

O UDCF foi proposto em:

> Ai, M. et al. **LBCF: A Large-Scale Budget-Constrained Causal Forest Algorithm.**
> Proceedings of the ACM Web Conference 2022 (WWW '22).

O código do algoritmo é dos autores do artigo e está disponível em:

> https://github.com/www2022paper/WWW-2022-PAPER-SUPPLEMENTARY-MATERIALS

**Este repositório não redistribui o código dos autores.** O notebook faz `git clone`
do repositório original em tempo de execução. O UDCF é implementado por modificação do
**GRF** (*Generalized Random Forests*, https://github.com/grf-labs/grf) e seus arquivos
são licenciados sob **GPL-3.0**, conforme os cabeçalhos presentes em cada arquivo-fonte.

Nenhum dos arquivos centrais do algoritmo é alterado. O único arquivo C++ escrito aqui é
o `main.cpp` — o ponto de entrada do programa, que carrega dados, indica quais colunas são
desfecho e tratamento, chama o treinador dos autores e grava as predições. A ele são
acrescentados apenas blocos de diagnóstico **somente-leitura**, que contam e imprimem
propriedades da floresta já treinada (número de nós internos, árvores sem split, e
frequência de splits por variável via `SplitFrequencyComputer`, classe que já existe no
código dos autores).

## Escopo

Cobre apenas a **primeira etapa** do artigo — estimação de CATE pelo UDCF. A segunda etapa
(otimização de orçamento via DGB / MCKP) está fora do escopo.

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `UDCF_Hillstrom_TCC.ipynb` | **Notebook canônico.** Roda tudo, do clone ao resultado final. |
| `TCC versão atual.docx` | Texto do TCC em andamento. |
| `UDCF_Hillstrom_Colab_*.ipynb` | Versões anteriores, mantidas para histórico. |
| `udcf_hillstrom/`, `udcf_hillstrom_chatGPT_corrigido.py` | Protótipos iniciais em Python, substituídos pelo uso do código C++ original. Mantidos apenas como registro. |

## Desenho experimental do notebook

O notebook executa quatro etapas, variando uma coisa por vez:

| Etapa | O que roda | Função |
|---|---|---|
| **0** | Código dos autores nos **dados dos autores** | Controle: valida que o pipeline e a compilação estão corretos e que a floresta produz splits na base original. |
| **1** | Mesmo binário, **dados Hillstrom**, hiperparâmetros default | Resultado principal. Em relação à Etapa 0, só os dados mudam. |
| **2** | Diagnóstico numérico do critério de split | Explica por que as Etapas 0 e 1 divergem. |
| **3** | Hillstrom com `imbalance_penalty = 0` | Sensibilidade a um hiperparâmetro (não ao algoritmo). |

## Execução

Feito para rodar no **Google Colab** (o UDCF exige toolchain C++). Abrir
`UDCF_Hillstrom_TCC.ipynb` no Colab e executar as células em ordem. O clone do
repositório dos autores, a instalação do `cmake`/`g++` e a compilação são feitos pelo
próprio notebook.

## Dados

- **Hillstrom**: base pública (MineThatData E-Mail Analytics Data Mining Challenge, 2008),
  64.000 clientes, baixada pelo notebook em tempo de execução.
- **Base RCT dos autores**: acompanha o repositório deles (~2.000 amostras criptografadas).
  O README dos autores é explícito: os dados são disponibilizados **"NOT FOR REPRODUCE THE
  RESULT"** — servem para validar o funcionamento do código, não para reproduzir os números
  do artigo, que é exatamente o uso feito na Etapa 0.
