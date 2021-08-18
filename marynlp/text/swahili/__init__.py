from typing import List
from .processor import SwahiliTextTransformer
import warnings

from ..formatter import normalize_text
from ... import funcutils as f

# NOTE: this is the old swahili text transformer (as a text processor)
tf = SwahiliTextTransformer()


def mask_by_rule(text: str):
    """Masks the text by observing the rules"""
    return tf(text)


def tokenize_by_rule(text: str) -> List[str]:
    """Tokenizes the texts by observing the swahili rules
    NOTE: you might want to check the `.tokenize` text method. 
        It's not doing it properly
    """
    warnings.warn("This is not working as expected. Don't use this unless you know what you are doing.")
    return tf.tokenize(text)


@f.apply(mask_by_rule)
def normalize_and_mask_text(text):
    """Normalize the swahili text"""
    return normalize_text(text)