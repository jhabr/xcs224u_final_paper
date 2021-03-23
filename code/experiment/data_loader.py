import os
from sklearn.model_selection import train_test_split

from baseline import ROOT
from utils.colors import ColorsCorpusReader
from utils.torch_color_describer import create_example_dataset


class DataLoader:

    def load_full_dataset(self):
        file_name = os.path.join(ROOT, "data", "colors", "filteredCorpus.csv")
        return self.__read_data(file_name, split=True)

    def load_bake_off_dataset(self):
        file_name = os.path.join(ROOT, "data", "colors", "cs224u-colors-bakeoff-data.csv")
        return self.__read_data(file_name, split=False)

    def load_dev_dataset_with_vocab(self, add_special_tokens=True, output_words=False):
        dev_colors, dev_text, dev_vocab = create_example_dataset(
            group_size=50,
            vec_dim=2,
            add_special_tokens=add_special_tokens,
            output_words=output_words
        )

        return [dev_vocab] + train_test_split(dev_colors, dev_text)

    def __read_data(self, file_name, split=False):
        corpus = ColorsCorpusReader(
            file_name,
            word_count=None,
            normalize_colors=True
        )

        examples = list(corpus.read())
        raw_colors, raw_texts = zip(*[[example.colors, example.contents] for example in examples])

        if split:
            return train_test_split(raw_colors, raw_texts)
        else:
            return raw_colors, raw_texts
