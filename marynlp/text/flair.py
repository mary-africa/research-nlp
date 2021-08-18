import re

from .. import funcutils as f

from . import swahili
from .swahili.ref import REGEX_UNK_TOKEN, REGEX_UNK_CHAR, REGEX_NUM_TOKEN

# this is here for when we might need it
_TOKEN_REPLACEMENT_PAIR = [
    (REGEX_UNK_TOKEN, '^'),
    (REGEX_UNK_CHAR, '#'),
    (REGEX_NUM_TOKEN, '$'),
]

# Information on the characters that are used for replacements
replacement_chars = [x[-1] for x in _TOKEN_REPLACEMENT_PAIR if x[0] != REGEX_UNK_TOKEN]

def _compatibility_replace_token_in_text(text: str):
    transformed = text
    for regex_val, sub_token in _TOKEN_REPLACEMENT_PAIR:
        transformed = re.sub(regex_val, sub_token, transformed)

    return transformed

@f.apply(_compatibility_replace_token_in_text)
def normalize_and_mask_swahili_text(text):
    """Normalize and mask swahili text and made flair compatible for training and inference"""
    return swahili.normalize_and_mask_text(text)
