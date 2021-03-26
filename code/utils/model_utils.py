import string

import torch
from transformers import (
    XLNetModel
)

import utils.utils as utils
from utils.utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL

__authors__ = "Anton Gochev, Jaro Habr, Yan Jiang, Samuel Kahn"
__version__ = "XCS224u, Stanford, Winter 2021"


def extract_input_embeddings(token_list, model, tokenizer, strip_punct=True, strip_symbols=True, add_special_tokens=False):
    """
    Parameters 
    ----------
    token_list: list of strings
        The colours description text in a list of strings. Expected is the raw format.
    
    model: huggingface transformer model
        Huggingface trasnformer model to be used for extracting embeddings.

    tokenizer: huggingface transformer tokenizer
        Huggingface trasnformer tokenizer to be used for generating the tokens and tokens ids.

    strip_punct: Boolean
        If set to True the punctuation will be stripped otherwise not. Default value is True.

    strip_symbols: Boolean
        If set to True the special symbols used by the models will be stripped otherwise not. 
        Default value is True.

    Returns
    -------
        A list of vectors that is the embeddings of the model. A list of token types that reperesnet
        the vocab.

    """

    embeddings = model.get_input_embeddings()
    model_embeddings = []
    model_vocab = []

    # add '' to the vocab and reserve a random vector at position 0. 
    # Needed for the padding in the model
    # model_vocab.append('')
    # model_embeddings.append(utils.randvec(1))

    for tokens in token_list:
        # if strip_punct:
        #     text = strip_punctuation(text)
        # input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=add_special_tokens)).unsqueeze(0)

        input_tokens = tokens[1:-1]  # remove <s>, </s>
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(input_tokens))
        # input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        vectors = embeddings(input_ids)

        for index, input_token in enumerate(input_tokens):
            if strip_symbols:
                input_token = strip_special_symbols(input_token)

            if input_token not in model_vocab:
                model_vocab.append(input_token)
                model_embeddings.append(vectors[index].detach().numpy())

    # add random vector at position 0 for '' padding. 
    # model_embeddings[0] = utils.randvec(len(model_embeddings[1]))
    model_vocab.insert(0, '')

    embeddings_dimension = len(model_embeddings[1])
    model_embeddings.insert(0, utils.randvec(embeddings_dimension))

    # add the special symbols and associated random vectors required for the model
    # to understand the end and start of utterences and an uknown vector
    # model_vocab.append(UNK_SYMBOL)
    # model_vocab.append(START_SYMBOL)
    # model_vocab.append(END_SYMBOL)

    # for i in range(3):
    #     model_embeddings.append(utils.randvec(embeddings_dimension))

    model_vocab.extend([UNK_SYMBOL, START_SYMBOL, END_SYMBOL])
    model_embeddings.extend([utils.randvec(embeddings_dimension)] * 3)

    assert len(model_vocab) == len(model_embeddings)

    return model_embeddings, model_vocab


def get_model_outputs(model, input_ids):
    """
    Parameters 
    ----------
    
    model: huggingface transformer model
        Huggingface trasnformer model to be used for determing the positions of tokens to be removed.

    input_ids

    Returns
    -------
        The model outputs in the format for the specific hugging face model.

    """

    if type(model) is XLNetModel:
        return model(input_ids=input_ids, decoder_input_ids=input_ids, output_hidden_states=True)
    else:
        return model(input_ids=input_ids, output_hidden_states=True)


def extract_colour_examples(examples, from_word_count=5):
    """
    Extracts all colour examples with more than 'from_word_count' words.

    Parameters 
    ----------
    examples: colors.ColorsCorpusExample
        A list of colour examples to subset from.

    from_word_count

    Returns
    -------
        A list of coulours examples.

    """
    result = []
    for row in examples:
        if row.contents.count(" ") > from_word_count:
            result.append(row)

    return result


def tokenize_colour_description(text, tokenizer, add_special_tokens=False):
    """
    Parameters 
    ----------
    text: string
        The input raw text colour string

    tokenizer
    add_special_tokens

    model: huggingface transformer model
        Huggingface trasnformer model to be used for extracting the tokens to be sequenced.

    Returns
    -------
        A list containing the tokenized sequence.

    """
    text = strip_punctuation(text.lower())
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=add_special_tokens)).unsqueeze(0)
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    result = [strip_special_symbols(input_token) for input_token in input_tokens]

    return [START_SYMBOL] + result + [END_SYMBOL]


def strip_punctuation(s):
    punc = s.maketrans(dict.fromkeys(string.punctuation))
    return s.translate(punc)


def strip_special_symbols(s):
    # symbols = ['_', '#', 'Ġ']
    # for symb in symbols:
    #     s = s.strip(symb)

    return s
