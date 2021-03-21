from baseline.model import BaselineColorEncoder
import utils.model_utils as mu

def evaluate(trained_model, tokenizer, color_seqs_test, texts_test, add_special_tokens=False):
    color_encoder = BaselineColorEncoder()
    tok_seqs = [mu.tokenize_colour_description(text, tokenizer, add_special_tokens) for text in texts_test]
    col_seqs = [color_encoder.encode_color_context(colors) for colors in color_seqs_test]

    return trained_model.evaluate(col_seqs, tok_seqs)