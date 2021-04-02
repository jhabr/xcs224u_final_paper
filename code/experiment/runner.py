from baseline.model import GloVeEmbedding
from experiment.experiment_library import ExperimentLibrary

if __name__ == '__main__':
    # baseline
    ExperimentLibrary.run_fourier_baseline(debug=False)
    ExperimentLibrary.run_fourier_baseline(hidden_dim=250, debug=False)
    ExperimentLibrary.run_fourier_baseline(embedding_dimension=GloVeEmbedding.DIM_300, debug=False)
    ExperimentLibrary.run_fourier_baseline_with_dropout(debug=False)
    ExperimentLibrary.run_vision_baseline(debug=False)
    ExperimentLibrary.run_vision_fourier_baseline(debug=False)

    # transformers
    ExperimentLibrary.run_fourier_bert_last_four_layers_sum(debug=False)
    ExperimentLibrary.run_vision_bert_last_four_layers_sum(debug=False)
    ExperimentLibrary.run_vision_fourier_bert_last_layer(debug=False)
    ExperimentLibrary.run_vision_fourier_electra_last_four_layers_sum(debug=False)
    ExperimentLibrary.run_vision_fourier_electra_second_last_layer(debug=False)
    ExperimentLibrary.run_vision_fourier_electra_concat_last_four_layers(debug=False)
