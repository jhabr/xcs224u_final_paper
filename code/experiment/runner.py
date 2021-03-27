from experiment.experiment_library import ExperimentLibrary

if __name__ == '__main__':
    # baseline
    # ExperimentLibrary.run_fourier_baseline(debug=True)
    # ExperimentLibrary.run_fourier_baseline_with_dropout(debug=True)
    # ExperimentLibrary.run_vision_baseline(debug=False)
    # ExperimentLibrary.run_vision_with_fourier_baseline(debug=True)

    # transformers
    ExperimentLibrary.run_fourier_bert_last_four_layers_sum(debug=False)
    # ExperimentLibrary.run_vision_bert_last_four_layers_sum(debug=False)
    # ExperimentLibrary.run_vision_fourier_bert_last_layer(debug=False)
    # ExperimentLibrary.run_vision_fourier_electra_last_four_layers_sum(debug=False)
