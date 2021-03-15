from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

from experiment.contextual_embeddings.model import ContextualDescriber
from utils.torch_color_describer import create_example_dataset
from utils.utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL


def create_dev_data():
    dev_color_seqs, dev_word_seqs, _ = create_example_dataset(
        group_size=50,
        vec_dim=2,
        add_special_tokens=False
    )

    dev_sentences = ["".join(dev_word_seq) for dev_word_seq in dev_word_seqs]

    return train_test_split(dev_color_seqs, dev_sentences)


if __name__ == '__main__':
    dev_colors_train, dev_colors_test, dev_sentences_train, dev_sentences_test = create_dev_data()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer.added_tokens_encoder = {
    #     '<s>': tokenizer.vocab_size + 1,
    #     '</s>': tokenizer.vocab_size + 1,
    #     '$UNK': tokenizer.vocab_size + 1
    # }
    # tokenizer.add_tokens([START_SYMBOL, END_SYMBOL, UNK_SYMBOL], special_tokens=True)
    # tokenizer.cls_token = START_SYMBOL
    # #tokenizer.cls_token_id = tokenizer.vocab["[CLS]"]
    # tokenizer.sep_token = END_SYMBOL
    # #tokenizer.sep_token_id = tokenizer.vocab["[SEP]"]
    # tokenizer.unk_token = UNK_SYMBOL
    # #tokenizer.unk_token_id = tokenizer.vocab["[UNK]"]
    model = BertModel.from_pretrained('bert-base-uncased')

    dev_contextual_model = ContextualDescriber(
        tokenizer=tokenizer,
        model=model,
        early_stopping=True
    )

    dev_contextual_model.fit(dev_colors_train, dev_sentences_train)

    print()
    print(dev_contextual_model.evaluate(dev_colors_test, dev_sentences_test))
