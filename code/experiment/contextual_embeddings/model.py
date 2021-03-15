import torch
from torch import nn
from transformers import PreTrainedTokenizer, PreTrainedModel
import numpy as np

from utils.torch_color_describer import Decoder, ContextualColorDescriber, Encoder, EncoderDecoder, ColorDataset
from utils.utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL


class ContextualEncoder(Encoder):
    """
    This class represents the system encode. It uses color representations extracted from pretrained models
    like ResNet18.
    """
    pass


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

    def get_embeddings(self, word_seqs, target_colors=None):
        _, repeats = word_seqs.size()

        embeddings = self.__extract_embeddings(word_seqs)

        target_colors_reshaped = torch.repeat_interleave(
            target_colors, repeats, dim=0
        ).view(*word_seqs.shape, -1)

        result = torch.cat((embeddings, target_colors_reshaped), dim=-1)

        expected = torch.empty(
            word_seqs.shape[0],
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

            # sequence = self.tokenizer.decode(ids)
            # inputs = self.tokenizer(sequence, add_special_tokens=False, output_hidden_states=True, return_tensors="pt")
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            embeddings.append(output.hidden_states[12].squeeze(0))

        return torch.stack(embeddings)


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
    the decoder in form of a ContextualEncoderDecoder class.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, early_stopping, freeze_embedding=False):
        self.tokenizer = tokenizer
        self.model = model
        # self.__update_vocab__()
        self.vocab = self.tokenizer.vocab

        super().__init__(
            vocab=tokenizer.vocab,
            embed_dim=768,
            early_stopping=early_stopping,
            freeze_embedding=freeze_embedding,
            # start_index=self.tokenizer.convert_tokens_to_ids(START_SYMBOL),
            # end_index=self.tokenizer.convert_tokens_to_ids(END_SYMBOL),
            # unk_index=self.tokenizer.convert_tokens_to_ids(UNK_SYMBOL)
            start_index=self.tokenizer.cls_token_id,
            end_index=self.tokenizer.sep_token_id,
            unk_index=self.tokenizer.unk_token_id
        )

    # def __update_vocab__(self):
    #     self.tokenizer.vocab[START_SYMBOL] = self.tokenizer.vocab_size + 1
    #     self.tokenizer.vocab[END_SYMBOL] = self.tokenizer.vocab_size + 1
    #     self.tokenizer.vocab[UNK_SYMBOL] = self.tokenizer.vocab_size + 1

    def build_dataset(self, color_seqs, word_seqs):
        self.color_dim = len(color_seqs[0][0])
        input_ids = []

        for sequence in word_seqs:
            seq_input_ids = self.tokenizer.encode(sequence, add_special_tokens=True)
            # seq_input_ids = [self.vocab[START_SYMBOL]] + seq_input_ids + [self.vocab[END_SYMBOL]]
            input_ids.append(seq_input_ids)

        # word_seqs = [[self.word2index.get(w, self.unk_index) for w in seq] for seq in word_seqs]

        ex_lengths = [len(input_id) for input_id in input_ids]

        return ColorDataset(color_seqs, input_ids, ex_lengths)

    def _convert_predictions(self, pred):
        # rep = []
        # for i in pred:
        #     i = i.item()
        #     rep.append(self.tokenizer.convert_ids_to_tokens(i))
        #     if i == self.end_index:
        #         return rep
        # return rep
        return self.tokenizer.decode(pred)

    def perplexities(self, color_seqs, word_seqs, device=None):
        probs = self.predict_proba(color_seqs, word_seqs, device=device)
        scores = []
        for prediction, sequence in zip(probs, word_seqs):
            s = np.array([t[w] for t, w in zip(prediction, self.tokenizer.encode(sequence, add_special_tokens=True))])
            scores.append(s)
        perp = [np.prod(s) ** (-1 / len(s)) for s in scores]
        return perp

        # for pred, seq in zip(probs, word_seqs):
        #     # Get the probabilities corresponding to the path `seq`:
        #     s = np.array([t[self.word2index.get(w, self.unk_index)] for t, w in zip(pred, seq)])
        #     scores.append(s)
        # perp = [np.prod(s) ** (-1 / len(s)) for s in scores]
        # return perp

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
            color_dim=self.color_dim,
            tokenizer=self.tokenizer,
            model=self.model
        )

        return ContextualEncoderDecoder(encoder, decoder)
