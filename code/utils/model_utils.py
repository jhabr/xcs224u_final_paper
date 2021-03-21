import torch
import string
from transformers import (
    BertTokenizer, BertModel,
    XLNetTokenizer, XLNetModel,
    ElectraTokenizer, ElectraModel,
    RobertaTokenizer, RobertaModel 
)
import utils.utils as utils
from utils.utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL

__authors__ = "Anton Gochev, Jaro Habr, Yan Jiang, Samuel Kahn"
__version__ = "XCS224u, Stanford, Winter 2021"

def extract_input_embeddings(colour_texts, model, tokenizer, strip_punct=True, strip_symbols=True, add_special_tokens=False):
    """
    Parameters 
    ----------
    colour_texts: list of strings
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
    result = dict()

    # add '' to the vocab and reserve a random vector at position 0. 
    # Needed for the padding in the model
    model_vocab.append('')
    model_embeddings.append(utils.randvec(1))
        
    for ct in colour_texts:
        if strip_punct:
            ct = strip_punctuation(ct)
        input_ids = torch.tensor(tokenizer.encode(ct, add_special_tokens=add_special_tokens)).unsqueeze(0)
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])        
        vectors = embeddings(input_ids)
        
        for i in range(len(input_tokens)):
            input_token = input_tokens[i]
            if strip_symbols:
                input_token = strip_special_symbols(input_token)

            if input_token not in model_vocab:
                model_vocab.append(input_token)
                model_embeddings.append(vectors[0][i].detach().numpy())

    # add random vector at position 0 for '' padding. 
    model_embeddings[0] = utils.randvec(len(model_embeddings[1]))

    # add the special symbols and associated random vectors required for the model
    # to understand the end and start of utterences and an uknown vector
    model_vocab.append(UNK_SYMBOL)
    model_vocab.append(START_SYMBOL)
    model_vocab.append(END_SYMBOL)

    for i in range(3):
        model_embeddings.append(utils.randvec(len(model_embeddings[1])))

    return model_embeddings, model_vocab

def get_model_outputs(model, input_ids):
    """
    Parameters 
    ----------
    
    model: huggingface transformer model
        Huggingface trasnformer model to be used for determing the positions of tokens to be removed.

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

    Returns
    -------
        A list of coulours examples.

    """
    result = []
    for row in examples:
        if row.contents.count(" ") > from_word_count:
            result.append(row)

    return result


def tokenize_colour_description(s, tokenizer, add_special_tokens=False):
    """
    Parameters 
    ----------
    
    s: string
        The input raw text colour string

    model: huggingface transformer model
        Huggingface trasnformer model to be used for extracting the tokens to be sequenced.

    Returns
    -------
        A list containing the tokenized sequence.

    """
    s = strip_punctuation(s.lower())
    input_ids = torch.tensor(tokenizer.encode(s, add_special_tokens=add_special_tokens)).unsqueeze(0)
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    result = []
    for i in range(len(input_tokens)):
        input_token = input_tokens[i]

        result.append(strip_special_symbols(input_token))

    return [START_SYMBOL] + result + [END_SYMBOL]


def strip_punctuation(s):
    punc = s.maketrans(dict.fromkeys(string.punctuation))
    return s.translate(punc)


def strip_special_symbols(s):
    # symbols = ['_', '#', 'Ġ']
    # for symb in symbols:
    #     s = s.strip(symb)
        
    return s