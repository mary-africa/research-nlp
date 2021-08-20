"""
NOTE: This is all OLD things
"""

from .base import token, compoundToken

class subword(token):    
    def __repr__(self):
        return "sb'{}'".format(self.extra_repr())

    def __hash__(self):
        return hash("sb-{}".format(str(self)))
    
class morpheme(subword):
    def __repr__(self):
        return "morph'{}'".format(self.extra_repr())

    def __hash__(self):
        return hash("mp-{}".format(str(self)))
    
class word(compoundToken):
    # character that combines subwords to represent a words
    separator = ""
    
    def __repr__(self):
        return "word('{}', l={})".format(str(self), len(self))
    
class sentence(compoundToken):
    # character that combines subwords to represent a words
    separator = " "
    
    def __repr__(self):
        return "sentence('{}', l={})".format(str(self), len(self))