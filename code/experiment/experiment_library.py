import time

from baseline.model import BaselineTokenizer, BaseColorEncoder, BaseEmbedding, BaselineDescriber, BaselineColorEncoder, \
    BaselineEmbedding
from experiment.data_preprocessor import DataPreprocessor, BaselineDataPreprocessor


class Experiment:
    def __init__(
            self,
            identifier: int,
            name: str,
            model_class: type,
            color_encoder: BaseColorEncoder,
            embedding: BaseEmbedding
    ):
        self.identifier = identifier
        self.name = name
        self.model_class = model_class
        self.color_encoder = color_encoder
        self.embedding = embedding

    def run(self, data_preprocessor: DataPreprocessor, debug=False, run_bake_off=True):
        print(f"-- Starting experiment {self.identifier}: {self.name}.")

        if debug:
            vocab, train_data, test_data = data_preprocessor.prepare_dev_data()
        else:
            vocab, train_data, test_data = data_preprocessor.prepare_training_data()

        created_embeddings, created_vocab = self.embedding.create_embeddings(vocab)

        model = self.model_class(
            embedding=created_embeddings,
            vocab=created_vocab,
            early_stopping=True
        )

        print("-- Training model...")
        start = time.time()
        model.fit(train_data[0], train_data[1])
        print(f"\n-- Training time: {(time.time() - start)} s")
        print("-- Evaluating model...")
        start = time.time()
        print(model.evaluate(test_data[0], test_data[0]))
        print(f"-- Evaluation time: {(time.time() - start)} s")

        if not debug and run_bake_off:
            colors, tokens = data_preprocessor.prepare_bake_off_data()
            print("-- Bake-Off...")
            start = time.time()
            print(model.evaluate(colors, tokens))
            print(f"-- Bake-Off time: {(time.time() - start)} s")

        print("-- Done experiment BASELINE.")


class ExperimentLibrary:

    @staticmethod
    def run_baseline(debug=False):
        experiment = Experiment(
            identifier=1,
            name="BASELINE",
            model_class=BaselineDescriber,
            color_encoder=BaselineColorEncoder(),
            embedding=BaselineEmbedding()
        )

        experiment.run(
            data_preprocessor=BaselineDataPreprocessor(),
            debug=debug,
            run_bake_off=True
        )
