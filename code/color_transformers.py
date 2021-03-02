import baseline_utils as bu
from transformers import PreTrainedModel
from transformers import BertConfig
import torch.nn as nn

"""
    class BertEncoder(nn.Module):
    
    class BertModel(BertPreTrainedModel):
    
    BertModel -> BertEncoder -> Config

"""


class ColorEncoder:
    pass


class ColorConfig(object):
    model_type: str = "bert"

    def __init__(self, hls_colors=None):
        self.hls_colors = hls_colors

    def to_dict(self):
        return {
            'hls_colors': self.hls_colors,
            'model_type': self.__class__.model_type
        }


class ColorTransformerModel(nn.Module):
    """
    Hope to use it with transformers encoder/decoder model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hls_colors = config.hls_colors

    def get_output_embeddings(self):
        return [
            bu.fourier_transform(bu.hls_to_hsv(hls_color)) for hls_color in self.hls_colors
        ]
