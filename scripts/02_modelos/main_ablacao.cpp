// Ponto de entrada para a base Hillstrom (NAO e arquivo do algoritmo).
// Uso: ./UDCF_RCT <min_node_size> <imbalance_penalty> <stabilize_splits 0|1> <entrada> <saida>
//
// stabilize_splits e parametro da propria funcao udcf_trainer dos autores:
//   true  -> UDCFSplittingRuleFactory           (Inter split + Intra split = UDCF)
//   false -> MultiRegressionSplittingRuleFactory (apenas Inter split = ablacao)
// Em ambos os casos o relabeling e o mesmo (UDCFRelabelingStrategy).

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
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
#include "analysis/SplitFrequencyComputer.h"
using namespace grf;

void update_predictions_file(const std::string& file_name,
                             const std::vector<Prediction>& predictions) {
  std::vector<std::vector<double>> values;
  values.reserve(predictions.size());
  for (const auto& prediction : predictions) {
    values.push_back(prediction.get_predictions());
  }
  FileTestUtilities::write_csv_file(file_name, values);
  std::cout << "predicoes gravadas em " << file_name << std::endl;
}

int main(int argc, char* argv[])
{
    uint min_node_size = 50;
    double imbalance_penalty = 0.01;
    bool stabilize_splits = true;
    std::string entrada = "/home/m229256/lbcf/Data/hillstrom/hillstrom_input.txt";
    std::string saida = "/home/m229256/lbcf/Code/Model/LBCF/output/hillstrom_pred";

    if (argc > 1) min_node_size = (uint) std::atoi(argv[1]);
    if (argc > 2) imbalance_penalty = std::atof(argv[2]);
    if (argc > 3) stabilize_splits = (std::atoi(argv[3]) != 0);
    if (argc > 4) entrada = argv[4];
    if (argc > 5) saida = argv[5];

    std::cout << "CONFIG num_trees=300 sample_fraction=0.5 mtry=3"
              << " min_node_size=" << min_node_size
              << " honesty=true honesty_fraction=0.5 prune=true alpha=0.05"
              << " imbalance_penalty=" << imbalance_penalty
              << " stabilize_splits=" << (stabilize_splits ? "true" : "false")
              << " num_threads=40 seed=42" << std::endl;
    std::cout << "REGRA " << (stabilize_splits
                 ? "UDCFSplittingRule (Inter + Intra)"
                 : "MultiRegressionSplittingRule (apenas Inter) [ABLACAO]") << std::endl;

    auto data_vec = load_data(entrada);
    Data data(data_vec);
    data.set_outcome_index(11);
    data.set_treatment_index({12, 13});

    size_t num_treatments = 2;

    ForestTrainer trainer = udcf_trainer(num_treatments, 1, stabilize_splits);
    ForestOptions options(300, 1, 0.5, 3, min_node_size, true, 0.5, true, 0.05,
                          imbalance_penalty, 40, 42, std::vector<size_t>(), 0);
    Forest forest = trainer.train(data, options);

    // ---------- diagnostico somente-leitura ----------
    std::cout << "DIAG_FOREST num_trees=" << forest.get_trees().size()
              << " num_variables=" << forest.get_num_variables() << std::endl;

    size_t total_nodes = 0, internal_nodes = 0, stump_trees = 0;
    for (const auto& tree : forest.get_trees()) {
      size_t tn = tree->get_child_nodes()[0].size();
      size_t internos = 0;
      for (size_t n = 0; n < tn; n++) {
        if (!tree->is_leaf(n)) {
          internos++;
        }
      }
      total_nodes += tn;
      internal_nodes += internos;
      if (internos == 0) {
        stump_trees++;
      }
    }
    std::cout << "DIAG_NODES total_nodes=" << total_nodes
              << " internal_nodes=" << internal_nodes
              << " stump_trees=" << stump_trees << std::endl;

    SplitFrequencyComputer freq_computer;
    std::vector<std::vector<size_t>> freq = freq_computer.compute(forest, 30);
    std::vector<size_t> por_var(forest.get_num_variables(), 0);
    size_t total_splits = 0;
    for (const auto& depth_counts : freq) {
      for (size_t v = 0; v < depth_counts.size(); v++) {
        por_var[v] += depth_counts[v];
        total_splits += depth_counts[v];
      }
    }
    std::cout << "DIAG_SPLIT_TOTAL " << total_splits << std::endl;
    std::cout << "DIAG_SPLIT_FREQ";
    for (size_t v = 0; v < por_var.size(); v++) {
      std::cout << " " << v << ":" << por_var[v];
    }
    std::cout << std::endl;
    // ---------- fim do diagnostico ----------

    ForestPredictor predictor = udcf_predictor(1, num_treatments, 1);
    std::vector<Prediction> predictions = predictor.predict_oob(forest, data, false);
    update_predictions_file(saida, predictions);

    return 0;
}
