from utils.torch_color_describer import create_example_dataset
from sklearn.model_selection import train_test_split
from baseline.model import BaselineEmbedding, BaselineDescriber


def create_dev_data():
    dev_color_seqs, dev_word_seqs, dev_vocab = create_example_dataset(
        group_size=50,
        vec_dim=2
    )

    return [dev_vocab] + train_test_split(dev_color_seqs, dev_word_seqs)


if __name__ == '__main__':
    dev_vocab, dev_colors_train, dev_colors_test, dev_tokens_train, dev_texts_test = \
        create_dev_data()

    embedding = BaselineEmbedding()

    dev_glove_embedding, dev_glove_vocab = embedding.create_embeddings(dev_vocab)

    dev_baseline_model = BaselineDescriber(
        dev_glove_vocab,
        embedding=dev_glove_embedding,
        early_stopping=True
    )

    dev_baseline_model.fit(dev_colors_train, dev_tokens_train)
