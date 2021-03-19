from enum import Enum

import torch

from baseline.model import BaseColorEncoder
import baseline.helper as bh


class ArchitectureType(Enum):
    ResNet18 = "resnet18"


class ConvolutionalColorEncoder(BaseColorEncoder):
    """
    This class is responsimble for loading HLS colors to other color formats.
    """

    def __init__(self, arch_type=ArchitectureType.ResNet18, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arch_type = arch_type
        self.model = None
        self.feature_extractor = None

    def __load_model_feature_extractor(self):
        self.model = torch.hub.load('pytorch/vision:v0.6.0', str(self.arch_type.value), pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])

    def __convert_to_imagenet_input(self, hls_color):
        rgb = bh.hls_to_rgb(hls_color)

        r = torch.full((224, 224), rgb[0]).unsqueeze(2)
        g = torch.full((224, 224), rgb[1]).unsqueeze(2)
        b = torch.full((224, 224), rgb[2]).unsqueeze(2)
        expanded_rep = torch.cat((r, g, b), 2)

        expanded_rep = expanded_rep.permute(2, 1, 0).unsqueeze(0)

        return expanded_rep

    def __extract_features_from_batch(self, examples):
        if self.feature_extractor is None:
            self.__load_model_feature_extractor()

        output = self.feature_extractor(examples)
        shape = output.shape
        output = output.reshape((shape[0], shape[1]))
        return output

    def encode_color_context(self, hls_colors):
        """
        Extracts colors from vision model.

        Parameters
        ----------
        hls_colors: list
            The color context in HLS format

        Returns
        -------
        list
            The extracted color representations.
        """

        image_embeddings = []

        with torch.no_grad():
            for hls_color in hls_colors:
                conv_input = self.__convert_to_imagenet_input(hls_color)
                conv_out = self.__extract_features_from_batch(conv_input)
                image_embeddings.append(conv_out)

        return image_embeddings
