import re

"""
Setting up the regular expression
for the application
"""

from typing import List, Iterable

# NOTE: *sighs* Adding this old code so that it works with the old layout of the code
#  I wouldn't use this like this. Frankly, this is abstraction is unnecessary

class DataTextTransformer(object):
    def extra_repr(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """This calls the transform()"""
        return self.transform(*args, **kwargs)

    def __repr__(self):
        return f'{type(self).__name__}(self.extra_repr())'


class StackedDataTextTransformer(DataTextTransformer):
    def __init__(self, transformers: List[DataTextTransformer]):
        assert len(transformers) > 0, 'You must include atleast one DataTextTransformer'

        for ix, trn in enumerate(transformers):
            assert isinstance(trn, DataTextTransformer) \
                , 'Expected transformer at index {} to be \'{}\', but got \'{}\'' \
                .format(ix, DataTextTransformer.__name__, type(trn).__name__)

        self.transformers = transformers

    def transform(self, text: str):
        t_text = text
        for trn in self.transformers:
            t_text = trn(t_text)

        return t_text

# -----------------------------------------------------------
from .data import token, Vocab
from typing import Dict, Any, List, Optional, Tuple

from itertools import chain

# from collections import OrderedDict

class Encoder(object):
    def __init__(self, encode_map: Dict[str, Any]):
        # You might want to consider using `OrderedDict` since dict doesn't guarantee the order
        self._encode_map = encode_map
        self._decode_map = { v: k for k, v in self._encode_map.items()}
    
    def encode(self, item: str):
        if item not in self._encode_map:
            raise KeyError("Item '%s' is not in items" % item)
            
        return self._encode_map[item]
    
    def decode(self, ix: int):
        if ix >= len(self):
            raise ValueError("Index %d is missing" % (ix))

        return self._decode_map[ix]
        
    def __len__(self):
        return len(self._encode_map)    

class BaseTokenEncoder(Encoder):
    def __init__(self, items: Iterable[token]):
        items = tuple(items) # converting to tuple
        super().__init__(dict(zip(items, range(len(items)))))

    def get_tokens(self):
        """Get the tokens"""
        return tuple(self._decode_map[ix] for ix in range(len(self)))
        
    def _proper_repr(self, key_value: Tuple[str, Any]):
        key, val = key_value
        return f"{key}={val}"
    
    def extra_repr(self):
        token_items = tuple(self._encode_map.items())
        
        if len(self) > 4:
            return ", ".join(list(map(self._proper_repr, token_items[:2]))) + "..., " + self._proper_repr(token_items[-1])
        
        return ", ".join(list(map(self._proper_repr, token_items)))
    
    def __repr__(self):
        return "Encoder(%s, l=%d)" % (self.extra_repr(), len(self))

    @property
    def size(self):
        return len(self)


class TokenEncoder(BaseTokenEncoder):
    def __init__(self, vocab: Vocab, special_tokens: List[token] = None, unk_token_idx: Optional[int] = None):
        self.special_tokens = [token('<UNK>')]
        self._unk_token_idx = 0
        
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self._unk_token_idx = unk_token_idx
            
        super().__init__(chain(self.special_tokens, vocab.get_tokens()))
        self.unk_token = self.special_tokens[self._unk_token_idx]

    def encode(self, item: token):# , enc_unk=True):
        try:
            return super().encode(item)
        except KeyError:
            # Return the token location for unknown token
            return super().encode(self.unk_token)

    def decode(self, ix):
        try:
            return super().decode(ix)
        except ValueError:
            return super().decode(self.unk_token)
    
    def __repr__(self):
        return "TokenEncoder(%s, l=%d)" % (self.extra_repr(), len(self))
        
class PaddedTokenEncoder(TokenEncoder):
    def __init__(self, vocab: Vocab):
        super().__init__(
            vocab, 
            special_tokens=[
                token('<PAD>'), token('<UNK>')
            ],
            unk_token_idx=1
        )

