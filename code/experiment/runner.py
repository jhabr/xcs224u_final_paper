from experiment.experiment_library import ExperimentLibrary

if __name__ == '__main__':
    experiment_library = ExperimentLibrary()

    # experiment_library.run_baseline(debug=False, run_bake_off=True)

    experiment_library.run_baseline(debug=True, decoder_drop_out=0.15, run_bake_off=True)
