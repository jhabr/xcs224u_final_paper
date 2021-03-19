import baseline.helper as bh
from enum import Enum
import os
import string
import re

from baseline import ROOT
from utils.torch_color_describer import Encoder, Decoder, EncoderDecoder, ContextualColorDescriber
import torch
import torch.nn as nn
import utils.utils as utils
from utils.utils import START_SYMBOL, END_SYMBOL
import colorsys
import pickle

__authors__ = "Anton Gochev, Jaro Habr, Yan Jiang, Samuel Kahn"
__version__ = "XCS224u, Stanford, Winter 2021"


class BaselineTokenizer:
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


class BaselineColorEncoder:
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

class ConvolutionalColorEncoder:
    """
    This class is responsimble for loading HLS colors to other color formats.
    """
    def __init__(self, arch_type, fourier_embeddings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arch_type = arch_type
        self.model = None
        self.feature_extractor = None
        self.fourier_embeddings = fourier_embeddings


    def _load_model_feature_extractor(self):
        self.model = torch.hub.load('pytorch/vision:v0.6.0', self.arch_type, pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])

    # Convert from HLS to RGB
    def _convert_color_to_rgb(self,color):
        rgb = colorsys.hls_to_rgb(color[0],color[1],color[2])
        return rgb

    def _convert_to_imagenet_input(self,hsl):
        rgb = self._convert_color_to_rgb(hsl)

        r = torch.full((224,224),rgb[0]).unsqueeze(2)
        g = torch.full((224,224),rgb[1]).unsqueeze(2)
        b = torch.full((224,224),rgb[2]).unsqueeze(2)
        expanded_rep = torch.cat((r,g,b),2)
        
        expanded_rep = expanded_rep.permute(2,1,0).unsqueeze(0)
        
        return expanded_rep

    # def _convert_color_tuple(self,colors):
    #     converted_colors = [[_convert_to_imagenet_input(col) for col in cols] for cols in colors ] 
    #     return converted_colors

    def _extract_features_from_batch(self, examples):
        if self.feature_extractor == None:
            self._load_model_feature_extractor()

        output = self.feature_extractor(examples)
        shape = output.shape
        output = output.reshape((shape[0],shape[1]))
        return output



    def encode_colors_from_convnet(self, hls_colors):
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
        # print(hls_colors)
        
        image_embeddings = []
        with torch.no_grad():
            for hls_color in hls_colors:
                conv_input = self._convert_to_imagenet_input(hls_color)
                conv_out = self._extract_features_from_batch(conv_input)
                if self.fourier_embeddings == True:
                    fourier_emb = torch.tensor(bh.fourier_transform(bh.hls_to_hsv(hls_color))).unsqueeze(0)
                    image_embeddings.append(torch.cat((fourier_emb,conv_out),dim=1))
                else:
                    image_embeddings.append(conv_out)

        return image_embeddings







class GloVeEmbedding(Enum):
    DIM_50 = 50
    DIM_100 = 100
    DIM_200 = 200
    DIM_300 = 300


class BaselineEmbedding:
    """
    This class is responsible for handling the embeddings of the baseline system.
    """
    GLOVE_HOME = os.path.join(ROOT, 'data', 'embeddings', 'glove.6B')

    def create_glove_embedding(self, vocab, dim=GloVeEmbedding.DIM_50):
        """
        Creates a GloVe embedding for the vocab with the selected dimension.

        Parameters
        ----------
        vocab: list of str
            Words to create embeddings for.

        dim: GloVeEmbedding
            The dimension for the glove embedding

        Returns
        -------
          embeddings
          expanded_vocab
        """
        glove_base_filename = f"glove.6B.{dim.value}d.txt"
        glove = utils.glove2dict(os.path.join(BaselineEmbedding.GLOVE_HOME, glove_base_filename))

        return utils.create_pretrained_embedding(glove, vocab)


class BaselineDecoder(Decoder):
    """
    This class represents the baseline system decoder.
    """

    def __init__(self, color_dim, *args, **kwargs):
        self.color_dim = color_dim
        super().__init__(*args, **kwargs)

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

    def build_graph(self):
        encoder = Encoder(
            color_dim=self.color_dim,
            hidden_dim=self.hidden_dim
        )

        decoder = BaselineDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim,
            color_dim=self.color_dim
        )

        return BaselineEncoderDecoder(encoder, decoder)
