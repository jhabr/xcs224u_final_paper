from experiment.experiment_library import ExperimentLibrary

if __name__ == '__main__':
    experiment_library = ExperimentLibrary()

    experiment_library.run_baseline(debug=True, run_bake_off=True)
