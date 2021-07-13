#!/usr/bin/env python
# coding: utf-8
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
        # print(ix, [v for v in self.oov.values() if v<28252])
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

class MorphemeEncoder(MorphologyAnalyzer):
    pass
    # def __init__(self,  morph_path=None, token_store_path=None):
    #     super(MorphemeEncoder, self).__init__(morph_path) 
    #     self.encoder = LabelEncoder
    #     self.break_dict = {}

    #     if token_store_path is not None:
    #         self.encoder = self.encoder(self.load_from_disk(token_store_path))

    # def fit(self, corpus = None, corpus_path = None, return_breaks=False, token_store_path=None, return_texts=False):
    #     '''
    #     Function to tokenize words based on morphemes collected from morphology analysis. 
    #     Uses the extracted morphemes as well as predefined rules to break words according to the 
    #     swahili syntax.
    #     Args:
    #         corpus      - text data to be tokeninzed
    #         corpus_path - path to text data on disk
    #     Returns:
    #         dictionary of each unique word in corpus and its corresponding list of morphemes
    #     '''

    #     if corpus_path is not None:
    #         corpus = self.read_from_path(corpus_path)

    #     if isinstance(corpus, str):
    #         corpus = corpus.split(' ')  

    #     text = None
    #     if return_texts:
    #         text = corpus 

    #     corpus = list(set([corp for corp in corpus if corp not in '']))

    #     break_dict = self.break_text(corpus, text)

    #     if not isinstance(self.encoder, LabelEncoder): 
    #         self.initialize_encoder(break_dict, token_store_path)

    #     if return_breaks:
    #         return break_dict

    # def initialize_encoder(self, break_dict, store_path):
    #     '''
    #     fit label encoder on training data
    #     '''
    #     tokens = []

    #     for v in break_dict.values():
    #         breaks = [v]
    #         tokens.extend(self.get_tokens(breaks))

    #     self.encoder = self.encoder(list(set(tokens)))

    #     if store_path is not None:
    #         self.save_to_disk(list(set(tokens)), store_path)

    # def fit_transform(self, corpus = None, corpus_path = None, token_store_path=None):
    #     '''
    #     get morphemes and corresponding label encodings
    #     Returns:
    #         list of arrays with integer representations of the tokens from the corpus
    #     '''
    #     if corpus_path is not None:
    #         corpus = self.read_from_path(corpus_path)

    #     assert corpus is not None, 'please pass in corpus or path to corpus'

    #     if isinstance(corpus, str):
    #         corpus = corpus.split(' ') 

    #     self.encoder = LabelEncoder

    #     self.fit(corpus=list(set(corpus)), token_store_path=token_store_path)
    #     break_dict = self.break_text(corpus)

    #     return self.get_encodings(break_dict)

    # def get_tokens(self, breaks):
    #     '''
    #     extract every token from list of list
    #     '''
    #     if isinstance(breaks[0], str) and not isinstance(breaks, str):
    #         tokens = breaks

    #     # elif 

    #     else:
    #         tokens = [tok for br in breaks for tok in br]

    #     return tokens

    # def encode(self, tokens):   
    #     '''
    #     label encoding of individual string
    #     '''
    #     return np.stack([np.array(self.encoder.encode(tok)) for tok in tokens])\
    #                  if not isinstance(tokens, str) \
    #                  else np.stack([np.array(self.encoder.encode(tokens))])

    # def get_encodings(self, breaks:dict):

    #     return list(map(self.encode, breaks.values()))

    # def transform(self, strings = None, string_path = None):
    #     '''
    #     label encoding of strings to be passed to the model
    #     '''
    #     if string_path is not None:
    #         strings = self.read_from_path(string_path)

    #     assert strings is not None, 'please pass strings or path to strings to be encoded'

    #     if isinstance(strings, str):
    #         strings = strings.split(' ')  

    #     # break_dict = self.fit(corpus=list(set(strings)), return_breaks=True)
    #     # assert self.break_dict, "tokenizer must be fit on training data"

    #     break_dict = self.break_text(strings)

    #     if isinstance(self.encoder, LabelEncoder):
    #         return self.get_encodings(break_dict)

    #     else:            
    #         self.initialize_encoder(break_dict)

    #         return self.get_encodings(break_dict)


    # def inverse_transform(self, arr):
    #     '''
    #     retrieve string from label-encoded tokens
    #     '''
    #     assert isinstance(self.encoder, LabelEncoder), 'encoder not initiated'

    #     return ''.join([self.encoder.decode(arr[i]) for i in range(len(arr))]) 