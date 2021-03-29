import abc


class BaseTokenizer(metaclass=abc.ABCMeta):
    """
    Base interface for tokenizers.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'encode') and
                callable(subclass.encode) or
                NotImplemented)

    @abc.abstractmethod
    def encode(self, text):
        raise NotImplementedError


class BaseColorEncoder(metaclass=abc.ABCMeta):
    """
    Base interface for color encoders.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'encode_color_context') and
                callable(subclass.encode_color_context) or
                NotImplemented)

    @abc.abstractmethod
    def encode_color_context(self, hls_colors):
        raise NotImplementedError


class BaseEmbedding(metaclass=abc.ABCMeta):
    """
    Base interface for embeddings.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'encode_color_context') and
                callable(subclass.encode_color_context) or
                NotImplemented)

    @abc.abstractmethod
    def create_embeddings(self, vocab):
        raise NotImplementedError
