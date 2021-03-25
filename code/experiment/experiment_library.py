import abc
import time

from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel, RobertaTokenizer, RobertaModel, \
    ElectraTokenizer, ElectraModel

from baseline.model import BaseEmbedding, BaselineDescriber, BaselineEmbedding
from datetime import datetime
from experiment.data_preprocessor import DataPreprocessor, BaselineDataPreprocessor, VisionBaselineDataPreprocessor, \
    VisionTransformerDataPreprocessor, TransformerDataPreprocessor
from experiment.model import TransformerEmbeddingDescriber, TransformerType, EmbeddingExtractorType
import utils.model_utils as mu


class TimeFormatter:
    @staticmethod
    def format(time_to_format):
        return time_to_format.strftime("%d/%m/%Y %H:%M:%S")


class Experiment:
    def __init__(
            self,
            identifier: int,
            name: str,
            model_class: type
    ):
        self.identifier = identifier
        self.name = name
        self.model_class = model_class
        self.model = None

    def run(self, data_preprocessor: DataPreprocessor, debug=False, run_bake_off=True):
        raise NotImplementedError

    def _train_model(self, colors_train, tokens_train):
        print("- 1. Training model...")
        start = time.time()
        self.model.fit(colors_train, tokens_train)
        print(f"\n-- Training time: {(time.time() - start)} s")

    def _evaluate_model(self, colors_test, tokens_test):
        print("- 2. Evaluating model...")
        start = time.time()
        print(self.model.evaluate(colors_test, tokens_test))
        print(f"-- Evaluation time: {(time.time() - start)} s")

    def _run_bake_off(self, colors, tokens):
        print("- 3. Bake-Off...")
        start = time.time()
        print(self.model.evaluate(colors, tokens))
        print(f"-- Bake-Off time: {(time.time() - start)} s")


class BaselineExperiment(Experiment):
    def __init__(self, embedding: BaseEmbedding, identifier: int, name: str, model_class: type, encoder_dropout=0.0,
                 decoder_dropout=0.0):
        super().__init__(identifier, name, model_class)
        self.embedding = embedding
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

    def run(self, data_preprocessor: DataPreprocessor, debug=False, run_bake_off=True):
        experiment_start = time.time()
        print(f"STARTING experiment {self.identifier}: {self.name}.\n"
              f"Start time: {TimeFormatter.format(datetime.now())}")

        if debug:
            vocab, colors_train, tokens_train, colors_test, tokens_test = data_preprocessor.prepare_dev_data()
        else:
            vocab, colors_train, tokens_train, colors_test, tokens_test = data_preprocessor.prepare_training_data()

        created_embeddings, created_vocab = self.__create_embeddings_vocab(vocab)
        self.model = self.__create_model(created_embeddings, created_vocab)
        assert self.model is not None

        self._train_model(colors_train, tokens_train)
        self._evaluate_model(colors_test, tokens_test)

        if not debug and run_bake_off:
            colors, tokens = data_preprocessor.prepare_bake_off_data()
            self._run_bake_off(colors, tokens)

        experiment_duration = time.time() - experiment_start
        print(f"DONE experiment {self.identifier}: {self.name}.\n"
              f"End time: {TimeFormatter.format(datetime.now())}. Duration: {experiment_duration} s.")

    def __create_embeddings_vocab(self, vocab):
        return self.embedding.create_embeddings(vocab)

    def __create_model(self, created_embeddings, created_vocab):
        return self.model_class(
            embedding=created_embeddings,
            vocab=created_vocab,
            early_stopping=True,
            encoder_dropout=self.encoder_dropout,
            decoder_dropout=self.decoder_dropout
        )


class TransformerExperiment(Experiment):
    def __init__(self, transformer_model: TransformerType, embeddings_extractor: EmbeddingExtractorType,
                 identifier: int, name: str, model_class: type, encoder_dropout=0.0, decoder_dropout=0.0):
        super().__init__(identifier, name, model_class)
        self.transformer_model = transformer_model
        self.embeddings_extractor = embeddings_extractor
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

    def run(self, data_preprocessor: DataPreprocessor, debug=False, run_bake_off=True):
        experiment_start = time.time()
        print(f"STARTING experiment {self.identifier}: {self.name}.\n"
              f"Start time: {TimeFormatter.format(datetime.now())}")

        if debug:
            vocab, colors_train, tokens_train, colors_test, tokens_test = data_preprocessor.prepare_dev_data()
        else:
            colors_train, tokens_train, colors_test, tokens_test = data_preprocessor.prepare_training_data()

        created_embeddings, created_vocab = self.__create_embeddings_vocab(tokens_train)
        self.model = self.__create_model(created_embeddings, created_vocab)
        assert self.model is not None

        self._train_model(colors_train, tokens_train)
        self._evaluate_model(colors_test, tokens_test)

        if not debug and run_bake_off:
            colors, tokens = data_preprocessor.prepare_bake_off_data()
            self._run_bake_off(colors, tokens)

        experiment_duration = time.time() - experiment_start
        print(f"DONE experiment {self.identifier}: {self.name}.\n"
              f"End time: {TimeFormatter.format(datetime.now())}. Duration: {experiment_duration} s.")

    def __create_embeddings_vocab(self, texts):
        model, tokenizer = self.__get_model_and_tokenizer()
        created_embeddings, created_vocab = mu.extract_input_embeddings(texts, model, tokenizer)
        return created_embeddings, created_vocab

    def __create_model(self, created_embeddings, created_vocab):
        return TransformerEmbeddingDescriber(
            vocab=created_vocab,
            embedding=created_embeddings,
            transformer=self.transformer_model,
            extractor=self.embeddings_extractor,
            early_stopping=True,
            batch_size=32
        )

    def __get_model_and_tokenizer(self):
        model, tokenizer = None, None

        if self.transformer_model == TransformerType.BERT:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            model = BertModel.from_pretrained('bert-base-cased')

        if self.transformer_model == TransformerType.XLNet:
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            model = XLNetModel.from_pretrained('xlnet-base-cased')

        if self.transformer_model == TransformerType.RoBERTa:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaModel.from_pretrained('roberta-base')

        if self.transformer_model == TransformerType.ELECTRA:
            tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
            model = ElectraModel.from_pretrained('google/electra-small-discriminator')

        return model, tokenizer


class ExperimentLibrary:

    @staticmethod
    def run_fourier_baseline(debug=False):
        experiment = BaselineExperiment(
            identifier=1,
            name="BASELINE: Fourier - GloVe",
            model_class=BaselineDescriber,
            embedding=BaselineEmbedding()
        )

        experiment.run(
            data_preprocessor=BaselineDataPreprocessor(),
            debug=debug,
            run_bake_off=True
        )

    @staticmethod
    def run_fourier_baseline_with_dropout(debug=False):
        experiment = BaselineExperiment(
            identifier=2,
            name="BASELINE:  Fourier - GloVe, Dropout = 0.15",
            model_class=BaselineDescriber,
            embedding=BaselineEmbedding(),
            encoder_dropout=0.15,
            decoder_dropout=0.15
        )

        experiment.run(
            data_preprocessor=BaselineDataPreprocessor(),
            debug=debug,
            run_bake_off=True
        )

    @staticmethod
    def run_vision_baseline(debug=False):
        experiment = BaselineExperiment(
            identifier=11,
            name="BASELINE: ResNet18 - GloVe",
            model_class=BaselineDescriber,
            embedding=BaselineEmbedding()
        )

        experiment.run(
            data_preprocessor=VisionBaselineDataPreprocessor(),
            debug=debug,
            run_bake_off=True
        )

    @staticmethod
    def run_vision_with_fourier_baseline(debug=False):
        experiment = BaselineExperiment(
            identifier=16,
            name="BASELINE: ResNet18 + Fourier - GloVe",
            model_class=BaselineDescriber,
            embedding=BaselineEmbedding()
        )

        experiment.run(
            data_preprocessor=VisionBaselineDataPreprocessor(fourier_embeddings=True),
            debug=debug,
            run_bake_off=True
        )

    @staticmethod
    def run_fourier_bert_last_layers_sum(debug=False):
        experiment = TransformerExperiment(
            identifier=28,
            name="TRANSFORMER: Fourier - Bert, Sum last layers",
            model_class=TransformerEmbeddingDescriber,
            transformer_model=TransformerType.BERT,
            embeddings_extractor=EmbeddingExtractorType.SUMLASTFOURLAYERS
        )

        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        experiment.run(
            data_preprocessor=TransformerDataPreprocessor(tokenizer=bert_tokenizer),
            debug=debug,
            run_bake_off=True
        )

    @staticmethod
    def run_vision_bert_last_layers_sum(debug=False):
        experiment = TransformerExperiment(
            identifier=29,
            name="TRANSFORMER: ResNet18 - Bert, Sum last layers",
            model_class=TransformerEmbeddingDescriber,
            transformer_model=TransformerType.BERT,
            embeddings_extractor=EmbeddingExtractorType.SUMLASTFOURLAYERS
        )

        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        experiment.run(
            data_preprocessor=VisionTransformerDataPreprocessor(
                tokenizer=bert_tokenizer,
                fourier_embeddings=False
            ),
            debug=debug,
            run_bake_off=True
        )

    @staticmethod
    def run_vision_fourier_bert_last_layers_sum(debug=False):
        experiment = TransformerExperiment(
            identifier=29,
            name="TRANSFORMER: ResNet18 + Fourier - Bert, Sum last layers",
            model_class=TransformerEmbeddingDescriber,
            transformer_model=TransformerType.BERT,
            embeddings_extractor=EmbeddingExtractorType.SUMLASTFOURLAYERS
        )

        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        experiment.run(
            data_preprocessor=VisionTransformerDataPreprocessor(
                tokenizer=bert_tokenizer,
                fourier_embeddings=False
            ),
            debug=debug,
            run_bake_off=True
        )
