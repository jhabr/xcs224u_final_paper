import os

from experiment.experiment_library import ExperimentLibrary


def set_python_path():
    if os.getenv('PYTHONPATH') is None:
        head, _ = os.path.split(os.path.abspath(os.getcwd()))
        os.environ['PYTHONPATH'] = head


if __name__ == '__main__':
    set_python_path()

    # baseline
    # ExperimentLibrary.run_fourier_baseline(debug=True)
    # ExperimentLibrary.run_fourier_baseline_with_dropout(debug=True)
    # ExperimentLibrary.run_vision_baseline(debug=False)
    # ExperimentLibrary.run_vision_with_fourier_baseline(debug=True)

    # transformers
    #ExperimentLibrary.run_fourier_bert_last_layers_sum(debug=False)
    #ExperimentLibrary.run_vision_bert_last_layers_sum(debug=True)
    ExperimentLibrary.run_vision_fourier_bert_second_last_layer(debug=False)

