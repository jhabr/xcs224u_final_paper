import time

from baseline.model import BaseEmbedding, BaselineDescriber, BaselineEmbedding
from experiment.data_preprocessor import DataPreprocessor, BaselineDataPreprocessor, VisionDataPreprocessor


class Experiment:
    def __init__(
            self,
            identifier: int,
            name: str,
            model_class: type,
            embedding: BaseEmbedding
    ):
        self.identifier = identifier
        self.name = name
        self.model_class = model_class
        self.embedding = embedding

    def run(self, data_preprocessor: DataPreprocessor, debug=False, run_bake_off=True):
        print(f"STARTING experiment {self.identifier}: {self.name}.")

        if debug:
            vocab, colors_train, tokens_train, colors_test, tokens_test = data_preprocessor.prepare_dev_data()
        else:
            vocab, colors_train, tokens_train, colors_test, tokens_test = data_preprocessor.prepare_training_data()

        created_embeddings, created_vocab = self.embedding.create_embeddings(vocab)

        model = self.model_class(
            embedding=created_embeddings,
            vocab=created_vocab,
            early_stopping=True
        )

        print("- 1. Training model...")
        start = time.time()
        model.fit(colors_train, tokens_train)
        print(f"\n-- Training time: {(time.time() - start)} s")
        print("- 2. Evaluating model...")
        start = time.time()
        print(model.evaluate(colors_test, tokens_test))
        print(f"-- Evaluation time: {(time.time() - start)} s")

        if not debug and run_bake_off:
            colors, tokens = data_preprocessor.prepare_bake_off_data()
            print("- 3. Bake-Off...")
            start = time.time()
            print(model.evaluate(colors, tokens))
            print(f"-- Bake-Off time: {(time.time() - start)} s")

        print(f"DONE experiment {self.identifier}: {self.name}.")


class ExperimentLibrary:

    @staticmethod
    def run_baseline(debug=False):
        experiment = Experiment(
            identifier=1,
            name="BASELINE - GloVe, Fourier",
            model_class=BaselineDescriber,
            embedding=BaselineEmbedding()
        )

        experiment.run(
            data_preprocessor=BaselineDataPreprocessor(),
            debug=debug,
            run_bake_off=True
        )

    @staticmethod
    def run_baseline_vision(debug=False):
        experiment = Experiment(
            identifier=11,
            name="VISION - GloVe, ResNet18",
            model_class=BaselineDescriber,
            embedding=BaselineEmbedding()
        )

        experiment.run(
            data_preprocessor=VisionDataPreprocessor(),
            debug=debug,
            run_bake_off=True
        )

    @staticmethod
    def run_baseline_vision_with_fourier(debug=False):
        experiment = Experiment(
            identifier=16,
            name="VISION - GloVe, ResNet18 + Fourier",
            model_class=BaselineDescriber,
            embedding=BaselineEmbedding()
        )

        experiment.run(
            data_preprocessor=VisionDataPreprocessor(fourier_embeddings=True),
            debug=debug,
            run_bake_off=True
        )
