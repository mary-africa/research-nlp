import pandas as pd
import numpy as np

from text.data.preprocessing import LabelEncoder, MorphemeEncoder
from text.data.structures import Document
from tqdm.notebook import tqdm

from typing import List

def collect_tokens(sent):
    for word in sent:
        for tok in word.values():
            if isinstance(tok, list):
                return tok
            else:
                return [tok]

def retrieve_text(tokenizer, document=None, document_path=None):
    assert document or document_path is not None, "please specify text from which to extract tokens"

    if document_path is not None:
        document = tokenizer.read_from_path(document_path)

    return sorted(list(set([d for d in document if d != ""])))

def retrieve_tokens(tokenizer, text):
    
    tok_dict = tokenizer.break_text(text)

    return [t for tok in tok_dict.values() for t in tok]#list(set())

def load_encoder(items, encode_unk=True, sort_items=False):
    return LabelEncoder(list(set(items)), encode_unk=encode_unk, sort_items=sort_items)

class Vocab():
    def __init__(self, document:list=None, document_path:str=None, data:pd.DataFrame=None, text_col:str=None, morpheme_template=None, token_store_path=None, label_encoder=None):
        super().__init__()

        self.morpheme_encoder = MorphemeEncoder(morpheme_template)
        
        if data is not None:
            assert text_col is not None, 'specify column with text'

            document = data[text_col].str.replace('[^\w\s]','', regex=True).str.lower().replace(np.nan, '').values.tolist()
            self.data = data
            self.glossary = sorted(list(set(' '.join(data[text_col].str.replace('[^\w\s]','', regex=True).str.lower().replace(np.nan, '').values.tolist()).split())))

        else:
            # unique words
            self.glossary = retrieve_text(self.morpheme_encoder, document, document_path)


        if token_store_path is not None:
            self.tokens = self.morpheme_encoder.load_from_disk(token_store_path)

        else:
            self.tokens: List[list] = [retrieve_tokens(self.morpheme_encoder, [lab]) for lab in tqdm(self.glossary)]

        if label_encoder is None:
            self.label_encoder = load_encoder(self.glossary, sort_items=True) 
        else:
            self.label_encoder = label_encoder
            self.label_encoder.encode_oov(self.glossary)
            
        self.document = Document(document_text=document, document_path=document_path, morph_path=morpheme_template)
        self.token_encoder = load_encoder([t for tok in self.tokens for t in tok], encode_unk=False, sort_items=True)
        self.start_ix = 0

    def __len__(self):
        return len(self.token_encoder.items)

    def __getitem__(self, ix):
        if isinstance(self.document.raw, str):
            return self.check_document(self.document, ix)

        return [self.check_document(document) for document in self.document[ix]] if not isinstance(ix, slice)\
             else [self.check_document(document) for i in range(ix.start, ix.stop) for document in self.document[i]]

    def check_document(self, document, ix=None):
        if ix is not None:
            if isinstance(ix, slice): 
                sentence = [s1 for i in range(ix.start, ix.stop) for s in range(len(document[i].words)) for s1 in document[i][s].values()]
            else:
                sentence = [s1 for s in range(len(document[ix].words)) for s1 in document[ix][s].values()]
        else:
            sentence = [s1 for s in range(len(document.words)) for s1 in document[s].values()]

        return self.get_indices([token for token in sentence if token!=""])


    def get_index(self, token):
        return self.token_encoder.encode(str(token))

    def get_indices(self, tokens):
        return [[self.get_index(token)] if isinstance(token, str) else [self.get_index(tok) for tok in token] for token in tokens]
    
    def save_to_file(self, path):
        self.morpheme_encoder.save_to_disk(self.tokens, path)

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_ix < len(self.document):
            tokens = self.__getitem__(self.start_ix)
            self.start_ix+=1
            return tokens
        raise StopIteration