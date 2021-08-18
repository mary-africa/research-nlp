"""
Contains the refence vocabulary 
that is used to contain information
that properly describes the information
"""

from typing import List
from overrides import overrides

UNK_TOKEN, REGEX_UNK_TOKEN = '[UNK]', r'\[UNK\]'
UNK_CHAR, REGEX_UNK_CHAR = '[UNKC]', r'\[UNKC\]'
NUM_TOKEN, REGEX_NUM_TOKEN = '[NUM]', r'\[NUM\]'

EOS_TOKEN, REGEX_EOS_TOKEN = '[EOS]', r'\[EOS\]'
BOS_TOKEN, REGEX_BOS_TOKEN = '[BOS]', r'\[BOS\]'
PAD_TOKEN, REGEX_PAD_TOKEN = '[PAD]', r'\[PAD\]'

class SwahiliRefenceVocabulary(object):
    """Banks the parts of words that make up swahili words

    This is to include the characters that make us valid swahili words
    Things to note:

    Normal Swahili words (includes):
    letters: a-z
    other characters: dash(-), apostrophe(')

    Other words (acceptable in swahili):
    letters:

    """

    # this number regex applies for currency, decimals, and normal numbers
    # modified version from:
    #   https://stackoverflow.com/questions/5917082/regular-expression-to-match-numbers-with-or-without-commas-and-decimals-in-text
    regex_for_numbers = r'(?<!\S)(?=.)(0|([1-9](\d*|\d{0,2}(,\d{3})*)))?(\.\d*[1-9])?(?!\S)|(\d+)'

    # list of allowed characters
    base_characters = 'abcdefghijklmnoprstuvwyz'
    base_numbers = '0123456789'
    base_word_non_letter_chars = '\'-'
    base_punctuations = '.,?!()%&/:[]'  # important to include '[' and ']'

    special_tokens = [UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, NUM_TOKEN, UNK_CHAR]
    regex_special_tokens = [REGEX_UNK_TOKEN, REGEX_EOS_TOKEN, REGEX_BOS_TOKEN, REGEX_NUM_TOKEN, REGEX_UNK_CHAR]

    _backslash_punctuations = [f'\\{c}' for c in base_punctuations]
    _non_word = _backslash_punctuations + [r'\s+']
    
    regex_non_word = "".join(_backslash_punctuations)
    regex_word = r'{}{}{}'.format(base_characters, base_numbers, base_word_non_letter_chars)
    
    inverse_regex_non_word = r'((?![{}{}]+)\W)'.format(base_word_non_letter_chars, "".join(_non_word))
    inverse_regex_word = r'((?![{}]+)[\w{}]+)'.format(regex_word, base_word_non_letter_chars)

    @classmethod
    def get_all_characters(cls):
        return cls.special_tokens + \
               list(cls.base_characters) + \
               list(cls.base_word_non_letter_chars) + \
               list(cls.base_numbers) + \
               list(cls.base_punctuations)
