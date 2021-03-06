import os
import re
import string
from enum import Enum

import torch
import torch.nn as nn

import baseline.helper as bh
import utils.utils as utils
from baseline import ROOT
from baseline.base import BaseTokenizer, BaseColorEncoder, BaseEmbedding
from utils.torch_color_describer import Encoder, Decoder, EncoderDecoder, ContextualColorDescriber
from utils.utils import START_SYMBOL, END_SYMBOL

__authors__ = "Anton Gochev, Jaro Habr, Yan Jiang, Samuel Kahn"
__version__ = "XCS224u, Stanford, Winter 2021"


class BaselineTokenizer(BaseTokenizer):
    """
    The BaselineTokenizer class is responsible for encoding/tokenizing the text
    for the baseline system.
    """

    def encode(self, text):
        """
        Encodes the text in order to be used with the baseline system.

        Parameters
        ----------
        text: string
            The text to be encoded.

        Returns
        -------
        list
            The encoded text as list of tokens
        """
        text = text.lower()
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

        endings = ['ish', 'er', 'est']
        tokens = []

        for token in text.split():
            for ending in endings:
                if token.endswith(ending):
                    token = re.sub(ending, '', token)
            tokens.append(token)

        return [START_SYMBOL] + tokens + [END_SYMBOL]


class BaselineColorEncoder(BaseColorEncoder):
    """
    This class is responsimble for encoding HLS colors to other color formats.
    """

    def encode_color_context(self, hls_colors):
        """
        Encodes HLS colors to HSV colors and performs a discrete fourier transform.

        Parameters
        ----------
        hls_colors: list
            The color context in HLS format

        Returns
        -------
        list
            The HSV-converted and fourier transformed colors
        """

        return [
            bh.fourier_transform(bh.hls_to_hsv(hls_color)) for hls_color in hls_colors
        ]


class GloVeEmbedding(Enum):
    DIM_50 = 50
    DIM_100 = 100
    DIM_200 = 200
    DIM_300 = 300


class BaselineEmbedding(BaseEmbedding):
    """
    This class is responsible for handling the embeddings of the baseline system.
    """
    GLOVE_HOME = os.path.join(ROOT, 'data', 'embeddings', 'glove.6B')

    def __init__(self, embedding_dimension=GloVeEmbedding.DIM_50):
        super(BaselineEmbedding, self).__init__()
        self.embedding_dimension = embedding_dimension

    def create_embeddings(self, vocab):
        """
        Creates a GloVe embedding for the vocab with the selected dimension.

        Parameters
        ----------
        vocab: list of str
            Words to create embeddings for.

        Returns
        -------
          embeddings
          expanded_vocab
        """
        glove_base_filename = f"glove.6B.{self.embedding_dimension.value}d.txt"
        glove = utils.glove2dict(os.path.join(BaselineEmbedding.GLOVE_HOME, glove_base_filename))
        created_embeddings, created_vocab = utils.create_pretrained_embedding(glove, vocab)

        return created_embeddings, created_vocab


class BaselineEncoder(Encoder):
    """
    This class represents the baseline system encoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rnn = nn.GRU(
            input_size=self.color_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )


class BaselineDecoder(Decoder):
    """
    This class represents the baseline system decoder.
    """

    def __init__(self, color_dim, dropout=0.0, *args, **kwargs):
        super().__init__(dropout=dropout, *args, **kwargs)
        self.color_dim = color_dim

        self.rnn = nn.GRU(
            input_size=self.embed_dim + self.color_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

    def get_embeddings(self, word_seqs, target_colors=None):
        _, repeats = word_seqs.size()

        embeddings = self.embedding(word_seqs)

        target_colors_reshaped = torch.repeat_interleave(
            target_colors, repeats, dim=0
        ).view(*word_seqs.shape, -1)

        result = torch.cat((embeddings, target_colors_reshaped), dim=-1)

        expected = torch.empty(
            word_seqs.shape[0],
            embeddings.shape[1],
            embeddings.shape[2] + target_colors.shape[1]
        )
        assert result.shape == expected.shape, \
            "Expected shape {}; got shape {}".format(expected.shape, result.shape)

        return result


class BaselineEncoderDecoder(EncoderDecoder):
    """
    The Encode-Decoder glue for the baseline system.
    """

    def forward(self, color_seqs, word_seqs, seq_lengths=None, hidden=None, targets=None):
        if hidden is None:
            hidden = self.encoder(color_seqs)

        target_colors = torch.stack([colors[-1] for colors in color_seqs])

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


class BaselineDescriber(ContextualColorDescriber):
    """
    Based on ContextualColorDescriber, this class bundles the encoder together with
    the decoder in form of a BaselineEncoderDecoder class.
    """

    def __init__(self, decoder_dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder_dropout = decoder_dropout
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def build_graph(self):
        encoder = BaselineEncoder(
            color_dim=self.color_dim,
            hidden_dim=self.hidden_dim
        )

        decoder = BaselineDecoder(
            color_dim=self.color_dim,
            dropout=self.decoder_dropout,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim
        )

        return BaselineEncoderDecoder(encoder, decoder)


class BaselineLSTMEncoder(Encoder):
    """
    This class represents the baseline encoder with a LSTM cell
    """

    def __init__(self, color_dim, hidden_dim):
        super().__init__(color_dim, hidden_dim)

        self.color_dim = color_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(
            input_size=self.color_dim,
            hidden_size=self.hidden_dim,
            batch_first=True)


class BaselineLSTMDecoder(BaselineDecoder):
    """
    This class represents the baseline system decoder with LSTM cell.
    """

    def __init__(self, color_dim, *args, **kwargs):
        self.color_dim = color_dim
        super().__init__(color_dim, *args, **kwargs)

        self.rnn = nn.LSTM(
            input_size=self.embed_dim + self.color_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )


class BaselineLSTMDescriber(ContextualColorDescriber):
    """
    Based on ContextualColorDescriber, this class bundles the LSTM encoder together with
    the LSTM decoder in form of a BaselineEncoderDecoder class.
    """

    def build_graph(self):
        encoder = BaselineLSTMEncoder(
            color_dim=self.color_dim,
            hidden_dim=self.hidden_dim
        )

        decoder = BaselineLSTMDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim,
            color_dim=self.color_dim
        )

        return BaselineEncoderDecoder(encoder, decoder)
