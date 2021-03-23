import time

from transformers import PreTrainedTokenizer

from baseline.model import BaselineTokenizer, BaselineColorEncoder, BaseColorEncoder
from experiment.data_loader import DataLoader
from experiment.vision import ConvolutionalColorEncoder
from utils.utils import UNK_SYMBOL
import utils.model_utils as mu


class DataPreprocessor:

    def __init__(self, color_encoder=None, tokenizer=None):
        self.color_encoder = color_encoder
        self.tokenizer = tokenizer
        self.data_loader = DataLoader()
        self.dev_dataset = self.data_loader.load_dev_dataset_with_vocab()
        self.full_dataset = self.data_loader.load_full_dataset()
        self.bake_off_dataset = self.data_loader.load_bake_off_dataset()

    def prepare_dev_data(self):
        vocab, colors_train, colors_test, tokens_train, tokens_test = self.dev_dataset
        return vocab, colors_train, tokens_train, colors_test, tokens_test

    def prepare_training_data(self):
        raise NotImplementedError

    def prepare_bake_off_data(self):
        raise NotImplementedError

    def __check_attributes(self):
        if self.tokenizer is None or self.color_encoder is None:
            raise AttributeError("Tokenizer and/or color_encoder is None.")


class BaselineDataPreprocessor(DataPreprocessor):

    def __init__(self, color_encoder: BaseColorEncoder = BaselineColorEncoder()):
        super().__init__(color_encoder=color_encoder, tokenizer=BaselineTokenizer())

    def prepare_training_data(self):
        self.__check_attributes()

        raw_colors_train, raw_colors_test, raw_texts_train, raw_texts_test = self.full_dataset

        start = time.time()
        print("- Extracting color representations for training data...")
        colors_train = [self.color_encoder.encode_color_context(colors) for colors in raw_colors_train]
        print(f"\n-- Extraction time: {(time.time() - start)} s")
        tokens_train = [self.tokenizer.encode(text) for text in raw_texts_train]

        start = time.time()
        print("- Extracting color representations for test data...")
        colors_test = [self.color_encoder.encode_color_context(colors) for colors in raw_colors_test]
        print(f"\n-- Extraction time: {(time.time() - start)} s")
        tokens_test = [self.tokenizer.encode(text) for text in raw_texts_test]

        vocab = sorted({word for tokens in tokens_train for word in tokens})
        vocab += [UNK_SYMBOL]

        return vocab, colors_train, tokens_train, colors_test, tokens_test

    def prepare_bake_off_data(self):
        self.__check_attributes()

        raw_colors, raw_texts = self.bake_off_dataset
        colors = [self.color_encoder.encode_color_context(colors) for colors in raw_colors]
        tokens = [self.tokenizer.encode(text) for text in raw_texts]

        return colors, tokens


class VisionBaselineDataPreprocessor(BaselineDataPreprocessor):
    def __init__(self, fourier_embeddings=False):
        super().__init__(color_encoder=ConvolutionalColorEncoder(fourier_embeddings=fourier_embeddings))


class TransformerDataPreprocessor(DataPreprocessor):
    def __init__(self, tokenizer: PreTrainedTokenizer, color_encoder: BaseColorEncoder = BaselineColorEncoder()):
        super().__init__(color_encoder=color_encoder, tokenizer=tokenizer)
        self.dev_dataset = self.data_loader.load_dev_dataset_with_vocab(add_special_tokens=False, output_words=True)

    def prepare_training_data(self):
        raw_colors_train, raw_colors_test, raw_texts_train, raw_texts_test = self.full_dataset

        start = time.time()
        print("- Extracting color representations for training data...")
        colors_train = [self.color_encoder.encode_color_context(colors) for colors in raw_colors_train]
        print(f"\n-- Extraction time: {(time.time() - start)} s")
        tokens_train = [
            mu.tokenize_colour_description(text, self.tokenizer, add_special_tokens=True) for text in raw_texts_train
        ]

        start = time.time()
        print("- Extracting color representations for test data...")
        colors_test = [self.color_encoder.encode_color_context(colors) for colors in raw_colors_test]
        print(f"\n-- Extraction time: {(time.time() - start)} s")
        tokens_test = [
            mu.tokenize_colour_description(text, self.tokenizer, add_special_tokens=True) for text in raw_texts_test
        ]

        return colors_train, tokens_train, colors_test, tokens_test

    def prepare_bake_off_data(self):
        self.__check_attributes()

        raw_colors, raw_texts = self.bake_off_dataset
        colors = [self.color_encoder.encode_color_context(colors) for colors in raw_colors]
        tokens = [mu.tokenize_colour_description(text, self.tokenizer, add_special_tokens=True) for text in raw_texts]

        return colors, tokens


class VisionTransformerDataPreprocessor(TransformerDataPreprocessor):
    def __init__(self, tokenizer: PreTrainedTokenizer, fourier_embeddings=False):
        super().__init__(
            color_encoder=ConvolutionalColorEncoder(fourier_embeddings=fourier_embeddings),
            tokenizer=tokenizer
        )
