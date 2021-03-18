from experiment.model import (
    TransformerEmbeddingDecoder, 
    TransformerEmbeddingDescriber,
    EmbeddingExtractorType,
    EmbeddingExtractor
)
from baseline.model import BaselineColorEncoder
import utils.model_utils as mu

def train_many(colors_train, tokens_train, vocab, embeddings, model, tokenizer, extractors=None):
    assert extractors != None and type(extractors) is list, \
            "Expected a list of extractors but got something else"
    
    describers = []
    for extractor in extractors:
        describer = TransformerEmbeddingDescriber(
            vocab=vocab,
            embedding=embeddings,
            model=model,
            tokenizer=tokenizer,
            embed_extractor=extractor,
            early_stopping=True)
        describer.fit(colors_train, tokens_train)
        describers.append(describer)
        
    return describers


def evaluate(trained_model, tokenizer, color_seqs_test, texts_test):
    color_encoder = BaselineColorEncoder()
    tok_seqs = [mu.tokenize_colour_description(text, tokenizer) for text in texts_test]
    col_seqs = [color_encoder.encode_color_context(colors) for colors in color_seqs_test]

    return trained_model.evaluate(col_seqs, tok_seqs)


def evaluate_many(models, tokenizer, raw_colors_test, texts_test):
    assert models != None and type(models) is list, \
        "Expected a list of models but got something else"
    
    results = dict()
    for model in models:
        results[model.get_extractor().get_extractor_type()] = \
            evaluate(model, tokenizer, raw_colors_test, texts_test)
        
    return results