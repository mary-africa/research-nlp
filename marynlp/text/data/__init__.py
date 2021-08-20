
from typing import List, Optional, Union, Iterable


class token(str):
    def __new__(cls, o, *args, **kwargs):
        _str = str.__new__(cls,  o, *args, **kwargs)
        _str._o = o
        return _str

    def __repr__(self):
        return "t'%s'" % (self._o)

class byte(token):
    """Unit representation of an value"""
    def __repr__(self):
        return "b'%s'" % (self._o)

class morph(token):
    """token  object that represent subword"""
    def __repr__(self):
        return "m'%s'" % (self._o)

class word(token):
    """token object that prepresentes a word"""
    def __repr__(self):
        return "w'%s'" % (self._o)

# Vocabulary special for words
# -------------------
class Vocab(object):
    def __init__(self, tokens: Iterable[token]):
        self._tokens = list(tokens)
        
        # MAYBE: this might be a problem
        self._tokens.sort()

    def has(self, token: token):
        return token in self._tokens
    
    def get_tokens(self):
        return self._tokens

    def __iter__(self):
        return iter(self._tokens)
    
    def __len__(self):
        return len(self._tokens)
    
    def extra_repr(self):
        if len(self) > 4:
            return ", ".join(self._tokens[:2]) + "..., " + self._tokens[-1]
        
        return ", ".join(self._token)
    
    def __repr__(self):
        return "Vocab([%s], len=%d)" % (self.extra_repr(), len(self))
    
    @property
    def size(self):
        return len(self)
    
    @classmethod
    def from_file(cls, file_path):
        pass