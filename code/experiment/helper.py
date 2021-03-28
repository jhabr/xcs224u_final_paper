from baseline.model import (
    BaselineColorEncoder, 
    BaselineLSTMDescriber,
    BaselineDescriber
)
import utils.model_utils as mu

def create_colours_sets(color_encoder, raw_colors_train, raw_colors_test, raw_colors_test_bo):
    colors_train = [ color_encoder.encode_color_context(colors) for colors in raw_colors_train ]
    colors_test = [ color_encoder.encode_color_context(colors) for colors in raw_colors_test ]
    colors_bo = [ color_encoder.encode_color_context(colors) for colors in raw_colors_test_bo ]
    
    return colors_train, colors_test, colors_bo

def create_tokens_sets(tokenizer, texts_train, texts_test, texts_test_bo, add_special_tokens=True):    
    tokens_train = [ mu.tokenize_colour_description(text, tokenizer, add_special_tokens) for text in texts_train ]
    tokens_test = [ mu.tokenize_colour_description(text, tokenizer, add_special_tokens) for text in texts_test ]
    tokens_bo = [ mu.tokenize_colour_description(text, tokenizer, add_special_tokens) for text in texts_test_bo ]
    
    return tokens_train, tokens_test, tokens_bo

def run_hiddim_options(hidden_dims, start, end, vocab, embed, colors, tokens, unit='LSTM'):
    for dim in hidden_dims:
        model = None
        if unit is 'LSTM':
            model = BaselineLSTMDescriber(
                vocab,
                embedding=embed,
                early_stopping=True,
                hidden_dim=dim
            )
        if unit is 'GRU':
            model = BaselineDescriber(
                vocab,
                embedding=embed,
                early_stopping=True,
                hidden_dim=dim
            )
        _ = model.fit(colors['train'][start:end], tokens['train'][start:end])
        print("train " + str(dim) + " - " + str(model.evaluate(colors['test'], tokens['test'])))
        print("bake-off " + str(dim) + " - " + str(model.evaluate(colors['bo'], tokens['bo'])))