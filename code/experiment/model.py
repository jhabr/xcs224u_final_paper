import time
from enum import Enum

import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, BertModel,
    XLNetTokenizer, XLNetModel,
    ElectraTokenizer, ElectraModel,
    RobertaTokenizer, RobertaModel
)

from utils.torch_color_describer import (
    Encoder, Decoder,
    EncoderDecoder, ContextualColorDescriber
)


class TransformerType(Enum):
    BERT = 100
    XLNet = 101
    RoBERTa = 102
    ELECTRA = 103


class EmbeddingExtractorType(Enum):
    STATIC = 100
    POSITIONAL = 101
    LAYER12 = 112
    STATICANDLAYER12 = 113
    LASTFOURLAYERS = 114
    SUMLASTFOURLAYERS = 115
    LAYER11 = 116
    SUMALLLAYERS = 117


class TransformerEmbeddingEncoder(Encoder):
    """
    This class represents an encoder with a LSTM cell
    """

    def __init__(self, color_dim, hidden_dim):
        super().__init__(color_dim, hidden_dim)
        self.color_dim = color_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(
            input_size=self.color_dim,
            hidden_size=self.hidden_dim,
            batch_first=True)


class TransformerEmbeddingDecoder(Decoder):
    """
    This class represents the baseline system decoder.
    """

    def __init__(self, color_dim, vocab, transformer=TransformerType.BERT, extractor=EmbeddingExtractorType.STATIC,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.color_dim = color_dim
        self.vocab = vocab
        self.transformer = transformer
        self.extractor = extractor
        self.model, self.tokenizer = self.__select_model()

        # input_size based on single embeddings such as the static embeddings
        input_size = self.embed_dim + self.color_dim
        # input_size taking into account a combination of two embeddings (static and contextual)
        # with the same dimension
        if extractor == EmbeddingExtractorType.STATICANDLAYER12:
            input_size = self.embed_dim * 2 + self.color_dim
            # input_size taking into account the dimensions of the last four layers
        if extractor == EmbeddingExtractorType.LASTFOURLAYERS:
            input_size = self.embed_dim * 4 + self.color_dim
            # input_size taking into account the dimension of ELECTRA static embeddings (128) and
        # contextual embeddings (256)
        if transformer == TransformerType.ELECTRA:
            if extractor == EmbeddingExtractorType.LASTFOURLAYERS:
                input_size = self.embed_dim * 8 + self.color_dim
            if extractor == EmbeddingExtractorType.STATICANDLAYER12:
                input_size = self.embed_dim * 3 + self.color_dim
            else:
                input_size = self.embed_dim * 2 + self.color_dim

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

    def get_embeddings(self, token_indices_list, target_colors=None):
        _, repeats = token_indices_list.size()

        embeddings = self.__extract_embeddings(token_indices_list)

        target_colors_reshaped = torch.repeat_interleave(
            target_colors, repeats, dim=0
        ).view(*token_indices_list.shape, -1)

        result = torch.cat((embeddings, target_colors_reshaped), dim=-1).to(self.device)

        expected = torch.empty(
            token_indices_list.shape[0],
            embeddings.shape[1],
            embeddings.shape[2] + target_colors.shape[1]
        )
        assert result.shape == expected.shape, \
            "Expected shape {}; got shape {}".format(expected.shape, result.shape)

        return result

    def __select_model(self):
        model = None
        tokenizer = None

        if self.transformer == TransformerType.BERT:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            model = BertModel.from_pretrained('bert-base-cased')

        if self.transformer == TransformerType.XLNet:
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            model = XLNetModel.from_pretrained('xlnet-base-cased')

        if self.transformer == TransformerType.RoBERTa:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaModel.from_pretrained('roberta-base')

        if self.transformer == TransformerType.ELECTRA:
            tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
            model = ElectraModel.from_pretrained('google/electra-small-discriminator')

        return model, tokenizer

    def __extract_embeddings(self, token_indices_list):
        if self.extractor == EmbeddingExtractorType.STATIC:
            return self.embedding(token_indices_list)

        if self.extractor == EmbeddingExtractorType.POSITIONAL:
            return self.__extract_layer_embedding(token_indices_list, layer_index=0)

        if self.extractor == EmbeddingExtractorType.LAYER11:
            return self.__extract_layer_embedding(token_indices_list, layer_index=11)

        if self.extractor == EmbeddingExtractorType.LAYER12:
            return self.__extract_layer_embedding(token_indices_list, layer_index=12)

        if self.extractor == EmbeddingExtractorType.STATICANDLAYER12:
            last_layer_embeddings = self.__extract_layer_embedding(token_indices_list, layer_index=12)
            return self.__combine_layers_with_static_embeddings(token_indices_list, last_layer_embeddings)

        if self.extractor == EmbeddingExtractorType.LASTFOURLAYERS:
            return self.__combine_last_four_layers(token_indices_list)
            # return self.__combine_layers_with_static_embeddings(word_seqs, four_layers)

        if self.extractor == EmbeddingExtractorType.SUMLASTFOURLAYERS:
            return self.__sum_layers(token_indices_list, number_of_layers=4)

        if self.extractor == EmbeddingExtractorType.SUMALLLAYERS:
            return self.__sum_layers(token_indices_list, number_of_layers=12)

    def __combine_last_four_layers(self, word_seqs):
        # print('in combine')
        layer_12 = self.__extract_layer_embedding(word_seqs, layer_index=12)
        layer_11 = self.__extract_layer_embedding(word_seqs, layer_index=11)
        layer_10 = self.__extract_layer_embedding(word_seqs, layer_index=10)
        layer_9 = self.__extract_layer_embedding(word_seqs, layer_index=9)
        # print(layer_12.shape)
        # print(layer_11.shape)
        # print(layer_10.shape)
        # print(layer_9.shape)

        result = torch.cat((layer_9, layer_10, layer_11, layer_12), dim=-1)

        return result

    def __combine_layers_with_static_embeddings(self, word_seqs, last_layer_embeddings):
        return torch.cat((self.embedding(word_seqs), last_layer_embeddings), dim=-1)

    def __extract_layer_embedding(self, token_indices_list, layer_index):
        embeddings = []

        for token_indices in token_indices_list:
            tokens = []
            for token_index in token_indices:
                tokens.append(self.vocab[token_index])
            input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0).to(self.device)
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            embeddings.append(outputs.hidden_states[layer_index].squeeze(0))

        return torch.stack(embeddings)

    def __sum_layers(self, token_indices_list, number_of_layers):
        total_layers_count = 13  # we know that from the transformer architecture
        start_index = total_layers_count - number_of_layers
        embeddings = []

        for token_indices in token_indices_list:
            tokens = []
            for token_index in token_indices:
                tokens.append(self.vocab[token_index])
            input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0).to(self.device)
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            # result = outputs.hidden_states[start_index].squeeze(0)
            result = 0

            for layer_index in range(start_index, total_layers_count):
                result += outputs.hidden_states[layer_index].squeeze(0)
            embeddings.append(result)

        return torch.stack(embeddings)


class TransformerEmbeddingDescriber(ContextualColorDescriber):
    """
    Based on ContextualColorDescriber, this class extends with input parameters allowing 
    to set an emebdding extractor function, Huggingface transformers model and tokeniser 
    to be used for the extraction.
    """

    def __init__(self, transformer=TransformerType.BERT, extractor=EmbeddingExtractorType.STATIC, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transformer = transformer
        self.extractor = extractor

    def build_graph(self):
        encoder = Encoder(
            color_dim=self.color_dim,
            hidden_dim=self.hidden_dim
        )

        decoder = TransformerEmbeddingDecoder(
            vocab=self.vocab,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim,
            color_dim=self.color_dim,
            transformer=self.transformer,
            extractor=self.extractor
        )

        return TransformerEmbeddingEncoderDecoder(encoder, decoder)


class TransformerEmbeddingEncoderDecoder(EncoderDecoder):
    """
    The Encode-Decoder glue for the transformers embedding based system.
    """

    def forward(self, color_seqs, word_seqs, seq_lengths=None, hidden=None, targets=None):
        if hidden is None:
            hidden = self.encoder(color_seqs)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        target_colors = torch.stack([colors[-1] for colors in color_seqs])
        target_colors.to(device)

        output, hidden = self.decoder(
            word_seqs=word_seqs,
            seq_lengths=seq_lengths,
            hidden=hidden,
            target_colors=target_colors
        )

        if self.training:
            return output
        else:
            return output, hidden
