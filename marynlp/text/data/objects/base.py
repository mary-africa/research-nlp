from __future__ import annotations
# data
# -------------

# Anything that is here SHOULD deal with data as if they are immutable
from typing import Union, List, Iterable

# structures
# -----------------------------------------

class token(object):
    def __init__(self, obj: Union[str, token]):
        if isinstance(obj, token):
            obj = obj.get()
            
        assert isinstance(obj, str), "Obj must be string for this to work"
        self.o = obj
        
    def __hash__(self):
        """For serialization"""
        return hash(str(self))
    
    def get(self):
        return self.o
    
    def __str__(self):
        return self.o
    
    def __eq__(self, other: token):
        return self.get() == other.get()

    def __ge__(self, other: token):
        return self.get() >= other.get()

    def __gt__(self, other: token):
        assert isinstance(other, token), "This is not `token` object: %s" % other
        return self.get() > other.get()

    def __le__(self, other: token):
        return self.get() <= other.get()

    def __lt__(self, other: token):
        return self.get() < other.get()
    
    def extra_repr(self):
        return str(self)
    
    def __repr__(self):
        return "t'{}'".format(self.extra_repr())
        
class mask_token(object):
    def __init__(self, obj: Union[str, mask_token]):
        if isinstance(obj, mask_token):
            obj = obj.label
            
        assert isinstance(obj, str), "Obj must be string for this to work"
        self.label = obj
    
    def get(self):
        return self.label
    
    def __eq__(self, other: token):
        return self.get() == other.get()
    
    def extra_repr(self):
        return self.label

    def __str__(self):
        return "<{}>".format(self.extra_repr())

    def __repr__(self):
        return "<{}>".format(self.extra_repr())

    def __hash__(self):
        return hash("<{}>".format(self.label))
    

class compoundToken(object):
    separator = ""
    def __init__(self, token_list: Union[compoundToken, Iterable[str]]):
        if isinstance(token_list, compoundToken):
            token_list = token_list.tokens

        # assert isinstance(token_list, ), "Input must be an Iterable for this to work"
        self.tokens = tuple(token_list)
        
    def __len__(self):
        """Number of subwords"""
        return len(self.tokens)
    
    def __str__(self):
        return (self.separator).join(map(str, self.tokens))
    
    def __repr__(self):
        return "cp('{}', tks={})".format(str(self), len(self))
