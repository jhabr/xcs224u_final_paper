from experiment.experiment_library import ExperimentLibrary

if __name__ == '__main__':
    #ExperimentLibrary.run_baseline(debug=True)
    ExperimentLibrary.run_baseline_with_dropout(debug=True)
    #ExperimentLibrary.run_baseline_vision(debug=False)
    ExperimentLibrary.run_baseline_vision_with_fourier(debug=False)
