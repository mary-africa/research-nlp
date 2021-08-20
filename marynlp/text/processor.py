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

class Encoder(object):
    def __init__(self, items: Iterable[str]):
        items = tuple(items)
        self._encode_map = dict(zip(items, range(len(items))))
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
    
OOV_TOKEN = token('<UNK>')

class TokenEncoder(Encoder):
    def __init__(self, vocab: Vocab):
        super().__init__(vocab.get_tokens())
        
#         self.oov = {} if oov is None else {it:len(self.items)+i for i,it in enumerate(oov) if it not in self.items}
#         self.ix = len(self.oov)
#         self.encode_unk = encode_unk
    def encode(self, item: token):# , enc_unk=True):
        try:
            return super().encode(item)
        except KeyError:
            return -1
#         item = item[0] if isinstance(item, list) else item

#         if not enc_unk or item in self.items:
#             return int(self.items.index(item)) if item in self.items else int(len(self.items)) 

#         if item in self.oov.keys():
#             return self.oov[item]
        
#         if self.encode_unk:
#             self.oov.update({item:int(len(self.items))+self.ix})
#             self.ix+=1

#             return self.oov[item]
        
#         return int(len(self.items))

#     def encode_oov(self, oov):
#         self.oov.update({it:len(self.items)+i for i,it in enumerate(oov) if it not in self.items})


    def decode(self, ix):
        try:
            return super().decode(ix)
        except ValueError:
            return OOV_TOKEN
#         if ix<len(self.items):
#             return self.items[int(ix)]
        
#         return [(k,v) for k,v in self.oov.items() if v==ix][0][0]