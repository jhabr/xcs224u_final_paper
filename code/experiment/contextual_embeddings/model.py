import torch
from torch import nn
from transformers import PreTrainedTokenizer, PreTrainedModel

from utils.torch_color_describer import Decoder, ContextualColorDescriber, Encoder, EncoderDecoder
from utils.utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL


class ContextualEncoder(Encoder):
    pass


class ContextualDecoder(Decoder):
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


class ContextualEncoderDecoder(EncoderDecoder):

    def __init__(self, encoder: ContextualEncoder, decoder: ContextualDecoder):
        super(ContextualEncoderDecoder, self).__init__(encoder, decoder)

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


class ContextualDescriber(ContextualColorDescriber):
    """
    Based on ContextualColorDescriber, this class bundles the encoder together with
    the decoder in form of a BaselineEncoderDecoder class.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, early_stopping, freeze_embedding=False):
        self.tokenizer = tokenizer
        self.model = model
        self.__update_vocab__()

        super().__init__(
            vocab=tokenizer.vocab,
            embed_dim=768,
            early_stopping=early_stopping,
            freeze_embedding=freeze_embedding,
            start_index=self.tokenizer.vocab[START_SYMBOL],
            end_index=self.tokenizer.vocab[END_SYMBOL],
            unk_index=self.tokenizer.vocab[UNK_SYMBOL]
        )

    def __update_vocab__(self):
        length = len(self.tokenizer.vocab.keys())
        self.tokenizer.vocab[START_SYMBOL] = length + 1
        self.tokenizer.vocab[END_SYMBOL] = length + 1
        self.tokenizer.vocab[UNK_SYMBOL] = length + 1

    def build_graph(self):
        encoder = ContextualEncoder(
            color_dim=self.color_dim,
            hidden_dim=self.hidden_dim
        )

        decoder = ContextualDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim,
            color_dim=self.color_dim
        )

        return ContextualEncoderDecoder(encoder, decoder)
