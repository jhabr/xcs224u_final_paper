import torch
import string
from transformers import (
    BertTokenizer, BertModel,
    XLNetTokenizer, XLNetModel,
    ElectraTokenizer, ElectraModel,
    RobertaTokenizer, RobertaModel 
)

from utils import START_SYMBOL, END_SYMBOL

__authors__ = "Anton Gochev, Jaro Habr, Yan Jiang, Samuel Kahn"
__version__ = "XCS224u, Stanford, Winter 2021"

def extract_input_embeddings(colour_texts, model, tokenizer, strip_punct=True, strip_symbols=True):
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
        A dictionannary containing the embeddings. A list containing the vocab.

    """

    embeddings = model.get_input_embeddings()
    result = dict()
        
    for ct in colour_texts:
        if strip_punct:
            ct = strip_punctuation(ct)
        input_ids = torch.tensor(tokenizer.encode(ct, add_special_tokens=False)).unsqueeze(0)
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])        
        vectors = embeddings(input_ids)
        
        for i in range(len(input_tokens)):
            input_token = input_tokens[i]
            if strip_symbols:
                input_token = strip_special_symbols(input_token)

            if input_token not in result:
                result[input_token] = vectors[0][i]
                    
    return result, list(result.keys())

def extract_contextual_embeddings(colour_texts, model, tokenizer, strip_punct=True, strip_symbols=True):
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
        A list containing the embeddings [token_<position>, vector] and a list containing the vocab [type of token].

    """

    embeddings = model.get_input_embeddings()
    result_embeddings = []
    result_vocab = dict()
        
    for ct in colour_texts:
        if strip_punct == True:
            ct = strip_punctuation(ct)
        input_ids = torch.tensor(tokenizer.encode(ct, add_special_tokens=False)).unsqueeze(0)
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])        
        outputs = get_model_outputs(model, input_ids)        
        vectors = outputs.hidden_states[0]
                
        for i in range(len(input_tokens)):
            if strip_symbols:
                input_token = strip_special_symbols(input_tokens[i])

            input_token = input_token + '_' + str(i)

            if input_token not in result_vocab:
                result_vocab[input_token] = input_token
                result_embeddings.append([input_token, vectors[0][i]])
                    
    return result_embeddings, list(result_vocab.keys())

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
    

def tokenize_colour_description(s, tokenizer):
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
    s = strip_punctuation(s)
    input_ids = torch.tensor(tokenizer.encode(s, add_special_tokens=False)).unsqueeze(0)
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    result = []
    for i in range(len(input_tokens)):
        result.append(strip_special_symbols(input_tokens[i]))

    return [START_SYMBOL] + result + [END_SYMBOL]


def strip_punctuation(s):
    punc = s.maketrans(dict.fromkeys(string.punctuation))
    return s.translate(punc)


def strip_special_symbols(s):
    symbols = ['_', '#', 'Ġ']
    for symb in symbols:
        s = s.strip(symb)
        
    return s