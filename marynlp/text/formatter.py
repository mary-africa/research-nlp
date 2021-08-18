"""
# reformatting
-----------------------
These are responsible for formatting the input texts
"""

import re

def lowercase(text: str) -> str:
    return text.lower()

def remove_punctuations(text: str) -> str:
    # if has ', only removes it.
    # so ng'ombe becomes ngombe
    text = text.replace("'", "")

    # text = text.replace('\ufeff', '')

    # replaces any other character that is not a a-zA-Z0-9 or \s
    #  to a space
    return re.sub(r"\W", " ", text)

def white_space_cleaning(text: str) -> str:
    """Removes the trailing white spaces """
    return text.strip()

"""------------------------------------------------"""
from unicodedata import normalize

_POSSIBLE_NORMAL_FORMS = ['NFC', 'NFKC', 'NFKD', 'NFD']
def normalize_text(text: str, normal_form: str = 'NFKC') -> str:
    """Normalize the text"""
    assert normal_form in _POSSIBLE_NORMAL_FORMS, "Unknown normal form '%s', Should be one among '%r'" % (normal_form, _POSSIBLE_NORMAL_FORMS)
    return normalize(normal_form, text)

