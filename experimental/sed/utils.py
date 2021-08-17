import torch
import numpy as np

import re
from typing import Iterable, List
from collections.abc import Callable

from torch.utils.data import Dataset

class MorphemeDataset(Dataset):
    def __init__(self, word_morph_list: Iterable[List[str]]):
        self.wml = list(word_morph_list)
        
    def __getitem__(self, index: int):
        return np.array(self.wml[index])
    
    def __len__(self):
        return len(self.wml)

class LMDataset(Dataset):
    def __init__(self, 
                 list_of_sentences: List[str],
                 word_tokenize: Callable):
        self.ls = list_of_sentences
        self.word_tokenize = word_tokenize

    def split_words(self, sentence: str) -> Iterable[str]:
        return re.split(r"\s+", sentence)

    def __len__(self):
        return len(self.ls)
    
    def __getitem__(self, index: int):
        return [self.word_tokenize(w) for w in self.split_words(self.ls[index])]

def morpheme_pad_collate(batch):
    x, x_len = [], []
    
    x_len = [m.shape[0] for m in batch]
    max_len = max(x_len)
    
    x_batch_ = [np.pad(t, (0, max_len - t.shape[0])) if t.shape[0] < max_len else t for t in batch]
    xx = torch.stack([torch.from_numpy(x.reshape(1,-1)).long() for x in x_batch_])

    return xx, torch.as_tensor(x_len).unsqueeze(dim=1)

def lm_pad_collate(batch):
    x, x_len = [], []
    
    x_len = [m.shape[0] for m in batch]
    max_len = max(x_len)
    
    x_batch_ = [np.pad(t, (0, max_len - t.shape[0])) if t.shape[0] < max_len else t for t in batch]
    xx = torch.stack([torch.from_numpy(x).long() for x in x_batch_])

    return xx, torch.as_tensor(x_len)
