import time
from baseline.model import BaselineTokenizer, BaselineColorEncoder, BaselineEmbedding, BaselineDescriber
from experiment.data_loader import DataLoader
from utils.utils import UNK_SYMBOL


class ExperimentLibrary:

    def __init__(self):
        self.data_loader = DataLoader()
        self.dev_dataset = self.data_loader.load_dev_dataset_with_vocab()
        self.full_dataset = self.data_loader.load_full_dataset()
        self.bake_off_dataset = self.data_loader.load_bake_off_dataset()

    def run_baseline(self, debug=False, run_bake_off=True):
        print("-- Starting experiment BASELINE.")
        tokenizer = BaselineTokenizer()
        color_encoder = BaselineColorEncoder()
        embedding = BaselineEmbedding()

        if debug:
            vocab, colors_train, colors_test, tokens_train, tokens_test = self.dev_dataset
        else:
            raw_colors_train, raw_colors_test, raw_texts_train, raw_texts_test = self.full_dataset
            colors_train = [color_encoder.encode_color_context(colors) for colors in raw_colors_train]
            tokens_train = [tokenizer.encode(text) for text in raw_texts_train]

            colors_test = [color_encoder.encode_color_context(colors) for colors in raw_colors_test]
            tokens_test = [tokenizer.encode(text) for text in raw_texts_test]

            vocab = sorted({word for tokens in tokens_train for word in tokens})
            vocab += [UNK_SYMBOL]

        glove_embedding, glove_vocab = embedding.create_glove_embedding(vocab)

        baseline_model = BaselineDescriber(
            glove_vocab,
            embedding=glove_embedding,
            early_stopping=True
        )

        print("-- Training model...")
        start = time.time()
        baseline_model.fit(colors_train, tokens_train)
        print(f"\n-- Training time: {(time.time() - start)} s")
        print("-- Evaluating model...")
        start = time.time()
        print(baseline_model.evaluate(colors_test, tokens_test))
        print(f"-- Evaluation time: {(time.time() - start)} s")

        if not debug and run_bake_off:
            raw_colors, raw_texts = self.bake_off_dataset
            colors = [color_encoder.encode_color_context(colors) for colors in raw_colors]
            tokens = [tokenizer.encode(text) for text in raw_texts]

            print("-- Bake-Off...")
            start = time.time()
            print(baseline_model.evaluate(colors, tokens))
            print(f"-- Bake-Off time: {(time.time() - start)} s")

        print("-- Done experiment BASELINE.")
