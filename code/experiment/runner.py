from experiment.data_preprocessor import DataPreprocessor
from experiment.experiment_library import ExperimentLibrary

if __name__ == '__main__':
    experiment_library = DataPreprocessor()

    # experiment_library.run_baseline(debug=False, run_bake_off=True)

    # ExperimentLibrary.run_baseline(debug=True)

    ExperimentLibrary.run_baseline_vision(debug=True)
