from baseline.model import BaselineTokenizer, BaselineColorEncoder, BaseColorEncoder
from experiment.data_loader import DataLoader
from experiment.vision import ConvolutionalColorEncoder
from utils.utils import UNK_SYMBOL


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


class BaselineDataPreprocessor(DataPreprocessor):

    def __init__(self, color_encoder: BaseColorEncoder = BaselineColorEncoder()):
        super().__init__(color_encoder=color_encoder, tokenizer=BaselineTokenizer())

    def prepare_training_data(self):
        self.__check_attributes()

        raw_colors_train, raw_colors_test, raw_texts_train, raw_texts_test = self.full_dataset

        colors_train = [self.color_encoder.encode_color_context(colors) for colors in raw_colors_train]
        tokens_train = [self.tokenizer.encode(text) for text in raw_texts_train]

        colors_test = [self.color_encoder.encode_color_context(colors) for colors in raw_colors_test]
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

    def __check_attributes(self):
        if self.tokenizer is None or self.color_encoder is None:
            raise AttributeError("Tokenier and/or color_encoder is None.")


class VisionDataPreprocessor(BaselineDataPreprocessor):
    def __init__(self):
        super().__init__(color_encoder=ConvolutionalColorEncoder())
