import re
from .ref import SwahiliRefenceVocabulary
from ..processor import DataTextTransformer

from .ref import UNK_TOKEN, NUM_TOKEN, UNK_CHAR
from typing import List


class SwahiliTextTransformer(DataTextTransformer):
    def __init__(self):
        # binding a lower case swahili reference vocabulary
        self.ref_vocab = SwahiliRefenceVocabulary()
        self.special_chars = self.ref_vocab.special_tokens

        self._regex_for_extraction = (
            r'[{}]+'.format(self.ref_vocab.regex_word),
            r'[{}]'.format(self.ref_vocab.regex_non_word),
        )
        self._regex_rule = "|".join(self.ref_vocab.regex_special_tokens + list(self._regex_for_extraction))
        self._regex_tokenizer = re.compile(self._regex_rule, flags=re.UNICODE | re.MULTILINE | re.DOTALL)

    @property
    def reference_vocabulary(self):
        return self.ref_vocab

    def valid_text_replace(self, text: str):
        # replace all words that dont follow swahili with '[UNK]'
        text = re.sub(self.ref_vocab.inverse_regex_word, UNK_TOKEN, text)

        # replace all non-allowed characters with [UNKC]
        text = re.sub(self.ref_vocab.inverse_regex_non_word, UNK_CHAR, text)

        # replace all number like patters with [NUM]
        text = re.sub(self.ref_vocab.regex_for_numbers, NUM_TOKEN, text)
        return text

    def tokenize(self, text: str) -> List[str]:
        return self._regex_tokenizer.findall(text)

    def transform(self, text: str) -> str:
        # convert to lower case
        text = text.lower()

        # replace swahili invalid words
        text = self.valid_text_replace(text)

        # form the string with model ready structure
        text = ' '.join(self.tokenize(text))

        return text
