import numpy as np
import torch
from torch import nn
from transformers import PreTrainedTokenizer, PreTrainedModel
import time

from utils.torch_color_describer import Decoder, ContextualColorDescriber, Encoder, EncoderDecoder, ColorDataset


class ContextualDecoder(Decoder):
    """
    This class represents the contextual system decoder.
    """

    def __init__(self, color_dim, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, *args, **kwargs):
        self.color_dim = color_dim
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.model = model

        self.rnn = nn.GRU(
            input_size=self.embed_dim + self.color_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

    def get_embeddings(self, input_ids, target_colors=None):
        _, repeats = input_ids.size()

        start = time.time()
        embeddings = self.__extract_embeddings(input_ids)
        print(f"\n-- embeddings loaded in: {(time.time() - start) * 1000} ms")

        target_colors_reshaped = torch.repeat_interleave(
            target_colors, repeats, dim=0
        ).view(*input_ids.shape, -1)

        result = torch.cat((embeddings, target_colors_reshaped), dim=-1)

        expected = torch.empty(
            input_ids.shape[0],
            embeddings.shape[1],
            embeddings.shape[2] + target_colors.shape[1]
        )
        assert result.shape == expected.shape, f"Expected shape {expected.shape}; got shape {result.shape}"

        return result

    def __extract_embeddings(self, input_ids):
        embeddings = []

        for ids in input_ids:
            input_ids = torch.LongTensor(ids).unsqueeze(0)
            attention_mask = torch.LongTensor([1] * len(ids)).unsqueeze(0)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            embeddings.append(output.hidden_states[12].squeeze(0))

        return torch.stack(embeddings)


class ContextualEncoderDecoder(EncoderDecoder):

    def __init__(self, encoder: Encoder, decoder: ContextualDecoder):
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
    the decoder in form of a ContextualEncoderDecoder class.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, early_stopping, freeze_embedding=False):
        self.tokenizer = tokenizer
        self.model = model
        self.vocab = self.tokenizer.vocab

        super().__init__(
            vocab=tokenizer.vocab,
            embed_dim=768,
            early_stopping=early_stopping,
            freeze_embedding=freeze_embedding,
            start_index=self.tokenizer.cls_token_id,
            end_index=self.tokenizer.sep_token_id,
            unk_index=self.tokenizer.unk_token_id
        )

    def build_dataset(self, color_seqs, word_seqs):
        self.color_dim = len(color_seqs[0][0])

        input_ids = [self.tokenizer.encode(sequence, add_special_tokens=True) for sequence in word_seqs]
        ex_lengths = [len(input_id) for input_id in input_ids]

        return ColorDataset(color_seqs, input_ids, ex_lengths)

    def _convert_predictions(self, pred):
        return self.tokenizer.decode(pred)

    def perplexities(self, color_seqs, word_seqs, device=None):
        probs = self.predict_proba(color_seqs, word_seqs, device=device)
        scores = []
        for prediction, sequence in zip(probs, word_seqs):
            s = np.array([t[w] for t, w in zip(prediction, self.tokenizer.encode(sequence, add_special_tokens=True))])
            scores.append(s)
        perp = [np.prod(s) ** (-1 / len(s)) for s in scores]
        return perp

    def build_graph(self):
        encoder = Encoder(
            color_dim=self.color_dim,
            hidden_dim=self.hidden_dim
        )

        decoder = ContextualDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim,
            color_dim=self.color_dim,
            tokenizer=self.tokenizer,
            model=self.model
        )

        return ContextualEncoderDecoder(encoder, decoder)
