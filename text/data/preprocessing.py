import numpy as np
from text.utils.morphology import MorphologyAnalyzer

class LabelEncoder:
    def __init__(self, items: list, sort_items = False, oov:list=None, encode_unk=True):
        self.items = items if not sort_items else sorted(items)
        self.oov = {} if oov is None else {it:len(self.items)+i for i,it in enumerate(oov) if it not in self.items}
        self.ix = len(self.oov)
        self.encode_unk = encode_unk

    def encode(self, item, enc_unk=True):
        item = item[0] if isinstance(item, list) else item

        if not enc_unk or item in self.items:
            return int(self.items.index(item)) if item in self.items else int(len(self.items)) 

        if item in self.oov.keys():
            return self.oov[item]
        
        if self.encode_unk:
            self.oov.update({item:int(len(self.items))+self.ix})
            self.ix+=1

            return self.oov[item]
        
        return int(len(self.items))

    def encode_oov(self, oov):
        self.oov.update({it:len(self.items)+i for i,it in enumerate(oov) if it not in self.items})

    def __call__(self, input):
        return self.encode(input)

    def decode(self, ix):
        if ix<len(self.items):
            return self.items[int(ix)]
        
        return [(k,v) for k,v in self.oov.items() if v==ix][0][0]

class OneHotEncoder:
    def __init__(self, items: list, sorted = False):
        self.items = items if not sorted else sorted(items)

    def encode(self, items):
        ohe = np.zeros(len(self.items)+1, dtype=int)
        for item in items:
            if item in self.items:
                ohe[int(self.items.index(item))]=1
            else:
                ohe[int(len(self.items))]

        return ohe

    def __call__(self, input):
        return self.encode(input)

    def decode(self, oh_arr):

        return [self.items[i] for i in np.nonzero(oh_arr)[0]]
