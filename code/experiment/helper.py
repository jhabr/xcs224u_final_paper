import numpy as np
from enum import Enum
import os
import utils.utils as utils

__authors__ = "Anton Gochev, Jaro Habr, Yan Jiang, Samuel Kahn"
__version__ = "XCS224u, Stanford, Winter 2021"


class EmbeddingType(Enum):
    BERT_TOKENS = 'bert/en.embed.bert.base_uncased.tokens.768d.txt'
    BERT = 'bert/en.embed.bert.base_uncased.vocab.768d.txt'
    XLNET = 'xlnet/en.embed.xlnet_base_cased.vocab.768d.txt'


class Embedding:
    """
    This class is responsible for creating an embedding dict from the selected embedding.

    Parameters
    ----------
    embedding : str
        Full path to the embedding file to be processed.
    """

    HOME = os.path.join('data', 'embeddings')

    def __init__(self, embedding: EmbeddingType = EmbeddingType.BERT):
        self.embedding: EmbeddingType = embedding

    def get_embeddings(self):
        """
        Embeddings vectors file reader.

        Returns
        -------
        dict
            Mapping words to their embedding vectors as `np.array`.

        """
        embedding_dict = {}
        file_name = os.path.join(Embedding.HOME, self.embedding.value)

        with open(file_name, encoding='utf8') as file:
            while True:
                try:
                    line = next(file).strip().split()
                    embedding_dict[line[0]] = np.array(line[1:], dtype=np.float)
                except StopIteration:
                    break
                except UnicodeDecodeError:
                    pass
        return embedding_dict

    def get_vocabulary_embeddings(self, vocabulary: list, required_tokens: tuple = ('$UNK', "<s>", "</s>")):
        """
        Returns the word embeddings for the given vocabulary, containing the required_tokens as well

        Parameters
        ----------
        vocabulary: list of str
            Words to create embeddings for.

        required_tokens: tuple of str
            Tokens that must have embeddings. If they are not available
            in the look-up, they will be given random representations.

        Returns
        -------
        np.array, list
            The np.array is an embedding for `vocab` and the `list` is
            the potentially expanded version of `vocab` that came in.

        """
        lookup = self.get_embeddings()

        dim = len(next(iter(lookup.values())))
        embeddings = np.array([lookup.get(w, utils.randvec(dim)) for w in vocabulary])

        for token in required_tokens:
            if token not in vocabulary:
                vocabulary.append(token)
                embeddings = np.vstack((embeddings, utils.randvec(dim)))
        return embeddings, vocabulary
