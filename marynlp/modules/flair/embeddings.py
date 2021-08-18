from typing import Callable, Union, List

from flair.embeddings import FlairEmbeddings
from flair.data import Sentence

from flair.embeddings import FastTextEmbeddings as FlairFastTextEmbeddings

from ... import funcutils as f
from ...text import flair

import re
from abc import abstractmethod

SWAHILI_LANGUAGE_CODE = 'sw'


class TokenEmbeddings(object):
    @abstractmethod
    def embed(self, text: Union[str, List[str]]):
        """Takes in a sentences, or list of sentences so that it can be converted into embeddings"""
        raise NotImplementedError()

    @abstractmethod
    def embedding_length(self):
        """To return the length of the embeddings"""
        raise NotImplementedError()


class SwahiliDynamicEmbeddings(TokenEmbeddings):
    def __init__(self, embedding_model: FlairEmbeddings):
        self.embeddings = embedding_model
        self.transform = flair.normalize_and_mask_swahili_text

    def embed(self, texts: Union[str, List[str]]) -> List[Sentence]:
        """removing the abstraction for Sentences"""
        if type(texts) == str:
            sentences = Sentence(self.transform(texts), language_code=SWAHILI_LANGUAGE_CODE)
        elif type(texts) == list:
            sentences = [Sentence(self.transform(text), language_code=SWAHILI_LANGUAGE_CODE) for text in texts]
        else:
            raise RuntimeError('Texts must be \'{}\' or \'{}\', instead got \'{}\''.format('str', 'List[str]', type(texts).__name__))

        self.embeddings.embed(sentences)
        return sentences

    def embedding_length(self) -> int:
        return self.embeddings.embedding_length

    @classmethod
    def load(cls, model_path: str, model_name = 'sw-flair-model'):
        model = FlairEmbeddings(model_path)
        model.name = model_name
        return cls(model)


class FastTextEmbeddings(TokenEmbeddings):
    """
    FastText as implemented and trained over the flair library

    NOTE: The implementation in here, is more correct
    """
    def __init__(self, embedding_model: FlairFastTextEmbeddings):
        self._emb = embedding_model
        self.process = f.apply(lambda text: re.sub(r'\'\-', '', text))(flair.normalize_and_mask_swahili_text)

    def embed(self, texts: Union[str, List[str]]):
        """Get embedding of words or words in a list"""
        if isinstance(texts, str):
            # working with the string
            text = self.process(texts)
            return self.word_embed(text)

        else:
            assert isinstance(texts, list), "Input must be a str or List[str]"
            texts = (map(texts, self._dtf))
            return list(map(texts, self.word_embed))

    def word_embed(self, text: str):
        """Get embedding of an individual word"""
        sent = Sentence(text)
        self._emb.embed(sent)

        return [(token.text, token.get_embedding()) for token in sent]

    def embedding_length(self) -> int:
        return self._emb.embedding_length

    @classmethod
    def load(cls, model_path: str):
        model = FlairFastTextEmbeddings(model_path)
        model.name = 'sw-fasttext-100'
        return cls(model)
