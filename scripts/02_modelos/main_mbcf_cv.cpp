// MBCF com validacao cruzada: treina num arquivo, prediz noutro.
// Configuracao do main.cpp original do CF_DT/MBCF_RCT:
//   instrumental_trainer(0.0, false), instrumento = tratamento
// Uso: ./MBCF_RCT <imbalance_penalty> <treino> <teste> <saida>

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
}

int main(int argc, char* argv[])
{
    double imbalance_penalty = 0.01;
    std::string treino = "treino.txt", teste = "teste.txt", saida = "saida.txt";
    if (argc > 1) imbalance_penalty = std::atof(argv[1]);
    if (argc > 2) treino = argv[2];
    if (argc > 3) teste = argv[3];
    if (argc > 4) saida = argv[4];

    auto data_vec = load_data(treino);
    Data data(data_vec);
    data.set_outcome_index(11);
    data.set_treatment_index(12);
    data.set_instrument_index(12);

    ForestTrainer trainer = instrumental_trainer(0.0, false);
    ForestOptions options(300, 1, 0.5, 3, 50, true, 0.5, true, 0.05,
                          imbalance_penalty, 40, 42, std::vector<size_t>(), 0);
    Forest forest = trainer.train(data, options);

    size_t internos = 0;
    for (const auto& tree : forest.get_trees()) {
      size_t tn = tree->get_child_nodes()[0].size();
      for (size_t n = 0; n < tn; n++) if (!tree->is_leaf(n)) internos++;
    }
    std::cout << "DIAG splits=" << internos << std::endl;

    auto data_vec2 = load_data(teste);
    Data data2(data_vec2);
    data2.set_outcome_index(11);
    data2.set_treatment_index(12);
    data2.set_instrument_index(12);

    ForestPredictor predictor = instrumental_predictor(5);
    std::vector<Prediction> predictions = predictor.predict(forest, data, data2, false);
    update_predictions_file(saida, predictions);
    std::cout << "gravado " << saida << std::endl;
    return 0;
}
