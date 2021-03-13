from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

from experiment.contextual_embeddings.model import ContextualDescriber
from utils.torch_color_describer import create_example_dataset


def create_dev_data():
    dev_color_seqs, dev_word_seqs, _ = create_example_dataset(
        group_size=50,
        vec_dim=2
    )

    dev_sentences = ["".join(dev_word_seq) for dev_word_seq in dev_word_seqs]

    return train_test_split(dev_color_seqs, dev_sentences)


if __name__ == '__main__':
    dev_colors_train, dev_colors_test, dev_sentences_train, dev_sentences_test = \
        create_dev_data()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    dev_contextual_model = ContextualDescriber(
        tokenizer=tokenizer,
        model=model,
        early_stopping=True
    )

    dev_contextual_model.fit(dev_colors_train, dev_sentences_train)
