import os

import numpy as np
from sklearn.model_selection import train_test_split

import utils.utils as utils
from baseline.model import (
    BaselineEmbedding
)
from utils.torch_color_describer import create_example_dataset
from experiment.word_embeddings.helper import Embedding, EmbeddingType


def create_dev_data():
    dev_color_seqs, dev_word_seqs, dev_vocab = create_example_dataset(
        group_size=50,
        vec_dim=2
    )

    dev_colors_train, dev_colors_test, dev_words_train, dev_words_test = \
        train_test_split(dev_color_seqs, dev_word_seqs)

    return dev_vocab, dev_colors_train, dev_words_train, dev_colors_test, dev_words_test


def read_glove():
    glove_base_filename = f"glove.6B.100d.txt"
    glove = utils.glove2dict(os.path.join(BaselineEmbedding.GLOVE_HOME, glove_base_filename))
    glove_vocab = list(glove.keys())
    print(len(glove_vocab))

    dim = len(next(iter(glove.values())))
    vocab = ["and", "is"]
    embedding = np.array([glove.get(w, utils.randvec(dim)) for w in vocab])

    for index, word in enumerate(vocab):
        print(f"word: {word}")
        print(embedding[index])
        print(f"len: {len(embedding[index])}")


def embeddings_head(embeddings):
    print(len(embeddings.keys()))
    for index, word in enumerate(embeddings.keys()):
        if index == 3:
            break
        print(f"{index}: {word} \n {embeddings[word]}")


def read_albert(vocab):
    albert = Embedding(EmbeddingType.ALBERT)
    albert_embeddings = albert.get_embeddings()
    albert_vocab_embeddings, vocab = albert.get_vocabulary_embeddings(vocabulary=vocab)
    embeddings_head(albert_embeddings)


def read_bert_tokens(vocab):
    bert = Embedding(EmbeddingType.BERT_TOKENS)
    bert_embeddings = bert.get_embeddings()
    bert_token_embeddings, vocab = bert.get_vocabulary_embeddings(vocabulary=vocab)
    embeddings_head(bert_embeddings)


def read_bert(vocab):
    bert = Embedding(EmbeddingType.BERT)
    bert_embeddings = bert.get_embeddings()
    bert_vocab_embeddings, vocab = bert.get_vocabulary_embeddings(vocabulary=vocab)
    embeddings_head(bert_embeddings)


def read_electra(vocab):
    electra = Embedding(EmbeddingType.ELECTRA)
    electra_embeddings = electra.get_embeddings()
    electra_vocab_embeddings, vocab = electra.get_vocabulary_embeddings(vocabulary=vocab)
    embeddings_head(electra_embeddings)


def read_elmo(vocab):
    elmo = Embedding(EmbeddingType.ELMO)
    elmo_embeddings = elmo.get_embeddings()
    elmo_vocab_embeddings, vocab = elmo.get_vocabulary_embeddings(vocabulary=vocab)
    embeddings_head(elmo_embeddings)


def read_xlnet(vocab):
    xlnet = Embedding(EmbeddingType.XLNET)
    xlnet_embeddings = xlnet.get_embeddings()
    bert_vocab_embeddings, vocab = xlnet.get_vocabulary_embeddings(vocabulary=vocab)
    embeddings_head(xlnet_embeddings)


def read_use(vocab):
    use = Embedding(EmbeddingType.USE)
    use_embeddings = use.get_embeddings()
    bert_vocab_embeddings, vocab = use.get_vocabulary_embeddings(vocabulary=vocab)
    embeddings_head(use_embeddings)


if __name__ == '__main__':
    dev_vocab, _, _, _, _ = create_dev_data()

    read_glove()

    read_albert(dev_vocab)
    read_bert_tokens(dev_vocab)
    read_bert(dev_vocab)
    read_electra(dev_vocab)
    read_elmo(dev_vocab)
    read_xlnet(dev_vocab)
    read_use(dev_vocab)
