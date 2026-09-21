# -*- coding: utf-8 -*-
# UDCF aplicado à base Hillstrom
# Pipeline completo para Google Colab:
# 1) clonar o repositório;
# 2) preparar a base Hillstrom;
# 3) ajustar hiperparâmetros para permitir splits;
# 4) adaptar o main.cpp;
# 5) compilar e executar o UDCF;
# 6) carregar e verificar os CATEs.

import shutil
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 0. Configurações
# ============================================================

REPO_URL = "https://github.com/www2022paper/WWW-2022-PAPER-SUPPLEMENTARY-MATERIALS.git"
ROOT = Path("/content/WWW-2022-PAPER-SUPPLEMENTARY-MATERIALS")

UDCF_ZIP = ROOT / "Code/Model/CF_DT/UDCF_RCT.zip"
UDCF_DIR = ROOT / "Code/Model/CF_DT/UDCF_RCT"
CORE_DIR = UDCF_DIR / "core"
BUILD_DIR = CORE_DIR / "build"

OPTIONS_PATH = CORE_DIR / "src/utilities/ForestTestUtilities.cpp"
MAIN_PATH = CORE_DIR / "main.cpp"

TRAIN_PATH = ROOT / "train_hillstrom_udcf.csv"
TEST_PATH = ROOT / "test_hillstrom_udcf.csv"
TEST_INDEX_PATH = ROOT / "test_hillstrom_indices.csv"

OUTPUT_PATH = ROOT / "Code/Model/CF_DT/output/UDCF_uplift_Hillstrom"

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)


def run(cmd, cwd=None):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


# ============================================================
# 1. Clonar o repositório e extrair o UDCF
# ============================================================

if ROOT.exists():
    shutil.rmtree(ROOT)

run(["git", "clone", REPO_URL, str(ROOT)])
run(["unzip", "-q", str(UDCF_ZIP), "-d", str(ROOT / "Code/Model/CF_DT")])

print("\nRepositório clonado e UDCF extraído.")


# ============================================================
# 2. Ler e preparar a base Hillstrom
# ============================================================

df_h = pd.read_csv(HILLSTROM_URL)

print("\nBase Hillstrom original:", df_h.shape)
print(df_h[[
    "recency", "history", "mens", "womens", "zip_code",
    "newbie", "channel", "segment", "conversion"
]].head())

df_udcf = df_h.copy()

df_udcf["zip_code_num"] = df_udcf["zip_code"].map({
    "Urban": 0,
    "Surburban": 1,
    "Rural": 2,
})

df_udcf["channel_num"] = df_udcf["channel"].map({
    "Phone": 0,
    "Web": 1,
    "Multichannel": 2,
})

# Tratamento multivalorado:
# No E-Mail     -> (0, 0) [controle]
# Mens E-Mail   -> (1, 0)
# Womens E-Mail -> (0, 1)
df_udcf["T_mens"] = (df_udcf["segment"] == "Mens E-Mail").astype(int)
df_udcf["T_womens"] = (df_udcf["segment"] == "Womens E-Mail").astype(int)

cols_udcf = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
    "zip_code_num",
    "channel_num",
    "conversion",
    "T_mens",
    "T_womens",
]

df_final = df_udcf[cols_udcf].copy()

if df_final.isna().any().any():
    raise ValueError(
        "Foram encontrados valores ausentes após a codificação. "
        "Verifique zip_code e channel."
    )

# Divisão estratificada pelos três grupos originais.
train_idx, test_idx = train_test_split(
    df_h.index,
    test_size=0.20,
    random_state=42,
    stratify=df_h["segment"],
)

train_h = df_final.loc[train_idx].copy()
test_h = df_final.loc[test_idx].copy()

# O C++ dos autores lê arquivos sem índice e sem cabeçalho.
train_h.to_csv(TRAIN_PATH, index=False, header=False, sep=" ")
test_h.to_csv(TEST_PATH, index=False, header=False, sep=" ")

# Guardar os índices originais para relacionar CATE ↔ cliente.
pd.DataFrame({"original_index": test_idx}).to_csv(TEST_INDEX_PATH, index=False)

print("\nFormato adaptado ao UDCF:", df_final.shape)
print(df_final.head())
print("\nTreino:", train_h.shape)
print("Teste :", test_h.shape)


# ============================================================
# 3. Ajustar os hiperparâmetros da floresta
# ============================================================

options_code = OPTIONS_PATH.read_text(encoding="utf-8")

replacements = {
    "uint min_node_size = 50;": "uint min_node_size = 5;",
    "double alpha = 0.05;": "double alpha = 0.01;",
    "double imbalance_penalty = 0.01;": "double imbalance_penalty = 0.0;",
}

for old, new in replacements.items():
    if old not in options_code:
        raise RuntimeError(
            "Não encontrei no código-fonte a configuração esperada:\n"
            + old
            + "\nO repositório pode ter mudado."
        )
    options_code = options_code.replace(old, new, 1)

OPTIONS_PATH.write_text(options_code, encoding="utf-8")

print(
    "\nHiperparâmetros ajustados: "
    "min_node_size=5, alpha=0.01, imbalance_penalty=0.0"
)


# ============================================================
# 4. Adaptar main.cpp para Hillstrom
# ============================================================

main_code = r"""
#include <iostream>
#include <string>
#include <unistd.h>

#include "tree/Tree.h"
#include "prediction/DefaultPredictionStrategy.h"
#include "commons/utility.h"
#include "forest/ForestPredictor.h"
#include "forest/ForestTrainer.h"
#include "utilities/FileTestUtilities.h"
#include "utilities/ForestTestUtilities.h"
#include "forest/ForestTrainers.h"
#include "forest/ForestPredictors.h"

using namespace grf;

void update_predictions_file(
    const std::string& file_name,
    const std::vector<Prediction>& predictions) {

  std::vector<std::vector<double>> values;
  values.reserve(predictions.size());

  for (const auto& prediction : predictions) {
    values.push_back(prediction.get_predictions());
  }

  FileTestUtilities::write_csv_file(file_name, values);
  std::cout << "success! predictions dump to "
            << file_name << std::endl;
}

int main() {

    char tmp[256];
    getcwd(tmp, 256);
    std::cout << "Current working directory: "
              << tmp << std::endl;

    auto data_vec = load_data(
        "/content/WWW-2022-PAPER-SUPPLEMENTARY-MATERIALS/"
        "train_hillstrom_udcf.csv"
    );

    Data data(data_vec);
    data.set_outcome_index(7);
    data.set_treatment_index({8, 9});

    auto data_vec2 = load_data(
        "/content/WWW-2022-PAPER-SUPPLEMENTARY-MATERIALS/"
        "test_hillstrom_udcf.csv"
    );

    Data data2(data_vec2);
    data2.set_outcome_index(7);

    // Mantém a convenção usada no main.cpp RCT dos autores.
    data2.set_treatment_index({8});

    size_t num_treatments = 2;

    ForestTrainer trainer =
        udcf_trainer(num_treatments, 1, true);

    ForestOptions options =
        ForestTestUtilities::default_options(true, 1);

    Forest forest = trainer.train(data, options);

    std::cout << "Numero de arvores = "
              << forest.get_trees().size()
              << std::endl;

    const auto& trees = forest.get_trees();

    for (size_t i = 0; i < 3 && i < trees.size(); i++) {
        std::cout << "Arvore " << i
                  << " numero de nos = "
                  << trees[i]->get_child_nodes()[0].size()
                  << std::endl;
    }

    ForestPredictor predictor =
        udcf_predictor(1, num_treatments, 1);

    std::vector<Prediction> predictions =
        predictor.predict(forest, data, data2, false);

    update_predictions_file(
        "/content/WWW-2022-PAPER-SUPPLEMENTARY-MATERIALS/"
        "Code/Model/CF_DT/output/UDCF_uplift_Hillstrom",
        predictions
    );

    return 0;
}
"""

# Backup do main original.
shutil.copy2(MAIN_PATH, MAIN_PATH.with_suffix(".cpp.original"))
MAIN_PATH.write_text(main_code, encoding="utf-8")

print("\nmain.cpp adaptado para Hillstrom.")


# ============================================================
# 5. Compilar do zero e executar o UDCF
# ============================================================

for item in BUILD_DIR.iterdir():
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()

run(["cmake", ".."], cwd=BUILD_DIR)
run(["make", "-j2"], cwd=BUILD_DIR)
run(["./UDCF_RCT"], cwd=BUILD_DIR)


# ============================================================
# 6. Ler e verificar os CATEs
# ============================================================

df_cate = pd.read_csv(OUTPUT_PATH, header=None)
df_cate.columns = ["CATE_Mens_Email", "CATE_Womens_Email"]

print("\nDimensão dos CATEs:", df_cate.shape)
print("\nNúmero de valores distintos por tratamento:")
print(df_cate.nunique())

if (df_cate.nunique() <= 1).any():
    raise RuntimeError(
        "As estimativas continuam constantes em pelo menos um tratamento. "
        "Verifique no log acima se as árvores possuem mais de 1 nó."
    )

print("\nPrimeiros CATEs estimados:")
print(df_cate.head(20))

# Relacionar novamente cada linha ao índice original do Hillstrom.
test_ids = pd.read_csv(TEST_INDEX_PATH)
df_cate_com_id = pd.concat(
    [test_ids.reset_index(drop=True), df_cate.reset_index(drop=True)],
    axis=1,
)

cate_with_id_path = ROOT / "CATE_Hillstrom_UDCF.csv"
df_cate_com_id.to_csv(cate_with_id_path, index=False)

print("\nCATEs heterogêneos obtidos com sucesso.")
print("Arquivo final:", cate_with_id_path)
