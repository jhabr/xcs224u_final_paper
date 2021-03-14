import torch
import torch.nn as nn
import string
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
from enum import Enum
import utils.utils as utils
from utils.utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL

class TransformerEmbeddingDecoder(Decoder):
    """
    This class represents the baseline system decoder.
    """

    def __init__(self, color_dim, vocab, model, tokenizer, embed_extractor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.color_dim = color_dim
        self.vocab = vocab                
        self.tokenizer = tokenizer
        self.model = model
        self.embed_extractor = embed_extractor
        
        self.rnn = nn.GRU(
            input_size=self.embed_dim + self.color_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

    def get_embeddings(self, word_seqs, target_colors=None):
        _, repeats = word_seqs.size()
        
        embeddings = self.embed_extractor.extract( 
            embedding=self.embedding,
            vocab=self.vocab, 
            word_seqs=word_seqs, 
            model=self.model, 
            tokenizer=self.tokenizer)

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


class TransformerEmbeddingDescriber(ContextualColorDescriber):
    """
    Based on ContextualColorDescriber, this class extends with input parameters allowing 
    to set an emebdding extractor function, Huggingface transformers model and tokeniser 
    to be used for the extraction.
    """

    def __init__(self, model, tokenizer, embed_extractor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.embed_extractor = embed_extractor
        self.tokenizer = tokenizer
        self.model = model

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
            tokenizer=self.tokenizer,
            model=self.model,
            embed_extractor=self.embed_extractor
        )

        return TransformerEmbeddingEncoderDecoder(encoder, decoder)


class TransformerEmbeddingEncoderDecoder(EncoderDecoder):
    """
    The Encode-Decoder glue for the transformers embedding based system.
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


class EmbeddingExtractorType(Enum):
    STATIC = 100
    POSITIONAL = 101
    LAYER12 = 112

class EmbeddingExtractor:

    def __init__(self, embed_extractor=EmbeddingExtractorType.STATIC):
        """
        extractor : EmbeddingExtractorType
            This is the enum specifying which extraction model to be applied. 
        """

        self.embed_extractor = embed_extractor

    def extract(self, embedding, vocab, word_seqs, model, tokenizer):
        """
        Extracts embeddings in the decoder get_embedding() method.

        Parameters
        ----------

        embedding : pytorch.nn.Embedding
            This is the embedding of the Describer model.

        vocab : list of strings
            This is the vocab used by the Describer model.

        word_seqs : torch.LongTensor
            This is a padded sequence, dimension (m, k), where k is
            the length of the longest sequence in the batch. The `forward`
            method uses `self.get_embeddings` to mape these indices to their
            embeddings.

        model : Huggingface transformer model
            This is the model to be used for the extraction of embeddings in the decoder.
            It is ignored if self.embed_extractor=EmbeddingExtractorType.STATIC

        tokenizer : Huggingface transformer tokenizer
            This is the tokenizer to be used for the extraction of embeddings in the decoder.
            It is ignored if self.embed_extractor=EmbeddingExtractorType.STATIC

        Returns
        -------
            The word_seqs embeddings to be used in the decoder. The shape is `(m, k, hidden_dim)`
            where m is the number of examples and k is the max lenght of the sequence 
            in the batch.

        """

        if self.embed_extractor == EmbeddingExtractorType.STATIC:
            return embedding(word_seqs)
        
        if self.embed_extractor == EmbeddingExtractorType.POSITIONAL:
            return __extract_layer_embedding(vocab, word_seqs, model, tokenizer, layer_index=0)

        if self.embed_extractor == EmbeddingExtractorType.LAYER12:
            return __extract_layer_embedding(vocab, word_seqs, model, tokenizer, layer_index=12)        

    def __extract_layer_embedding(self, vocab, word_seqs, model, tokenizer, layer_index):

        embeddings = []
        for ws in word_seqs:
            utterence = []
            for i in ws:
                utterence.append(self.vocab[i])                
            input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(utterence)).unsqueeze(0)
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            embeddings.append(outputs.hidden_states[layer_index].squeeze(0))

        embeddings = torch.stack(embeddings)

        return embeddings
        