import string
import re
from utils import START_SYMBOL, END_SYMBOL

__authors__ = "Anton Gochev, Jaro Habr, Yan Jiang, Samuel Kahn"
__version__ = "XCS224u, Stanford, Spring 2021"


class BaselineTokenizer:
  """
  The BaselineTokenizer class is responsible for encoding/tokenizing the text
  for the baseline system.
  """

  def encode(self, text):
    """
    Encodes the text in order to be used with the baseline system.

    :param text: string
        The text to be encoded.
    :return: list
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
