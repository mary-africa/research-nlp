from .base import token, mask_token
from .others import morpheme, subword, sentence

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