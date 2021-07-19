import torch
import string
import numpy as np
from pathlib import Path
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from .preprocessing import MorphemeEncoder as Tokenizer
from .vocab import Vocab

def pad_collate(batch, pretrained=False):
    if not pretrained:
        x, y1, y2, x_len = [], [], [], []
        for x_batch, y1_batch, y2_batch in batch:
            xx, yy, yz = [torch.from_numpy(x) for x in x_batch], y1_batch, y2_batch
            x_len.append([torch.as_tensor(len(x)).long().reshape(1,) for x in xx])

            xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).long()

            x.append(xx_pad)
            y1.append(torch.as_tensor(yy).long().reshape(1,))
            y2.append(torch.as_tensor(yz).long().reshape(1,))

        return x, x_len, torch.stack(y1), torch.stack(y2)  

    x, y1, y2 = [], [], []
    for x_batch, y1_batch, y2_batch in batch:
        xx, yy, yz = torch.from_numpy(x_batch).float(), y1_batch, y2_batch
        x_len = None

        x.append(xx)
        y1.append(torch.as_tensor(yy).long().reshape(1,))
        y2.append(torch.as_tensor(yz).long().reshape(1,))

    return torch.stack(x), x_len, torch.stack(y1), torch.stack(y2)  

def lm_pad_collate(batch, lang_mod, frozen_embeddings):
    x, y, x_len = [], [], []
    if not lang_mod:        
        for x_batch, y_batch in batch:
            _len = [torch.as_tensor(len(x)).long().reshape(1,) for x in x_batch]
            
            max_len = len(max(x_batch, key=len))
            x_batch = [np.pad(t, (0, max_len-t.shape[0])) if t.shape[0]<max_len else t[:max_len] for t in x_batch]
            xx = torch.stack([torch.from_numpy(x.reshape(1,-1)).long() for x in x_batch])
            
            x_len.append(_len)
            x.append(xx)
            y.append(torch.as_tensor(y_batch).long().reshape(1,))

        return x, torch.as_tensor(x_len), torch.stack(y)

    if frozen_embeddings:
        
        for x_batch, y_batch in batch:
            _len = torch.as_tensor(len(x_batch)).long().reshape(1,)
            # x_batch = torch.stack(x_batch)

            x_len.append(_len)
            x.append(x_batch)
            y.append(y_batch)

        return x, x_len, torch.stack(y)

    for x_batch, y_batch in batch:
        _len = [torch.as_tensor(len(x)).long().reshape(1,) for x in x_batch]
        x_batch = torch.stack([torch.as_tensor(np.pad(t, (0, max_len-t.shape[0]))) if t.shape[0]<max_len else t[:max_len] for t in x_batch])

        x_len.append(_len)
        x.append(x_batch)
        y.append(y_batch)

    return pad_sequence(x, batch_first=True), x_len, torch.stack(y)

    # x, y = [], []
    # for x_batch, y_batch in batch:
    #     xx = torch.from_numpy(x_batch).float()
    #     x_len = None

    #     x.append(xx)
    #     y.append(torch.as_tensor(y_batch).long().reshape(1,))

    # return torch.stack(x), x_len, torch.stack(y)


class SAMDataset(Dataset):
    def __init__(self, vocab, emotion_col, sentiment_col, freeze_embeddings=True):
        self.vocab = vocab
        self.emotion_col = emotion_col
        self.sentiment_col = sentiment_col
        self.frozen_mebeddings = freeze_embeddings
        
    def __len__(self):
        return len(self.vocab.document)

    def __getitem__(self, idx):
        if not self.frozen_mebeddings:
            tokens = self.vocab[idx]
            emotion, sentiment = self.vocab.data.iloc[idx][self.emotion_col], self.vocab.data.iloc[idx][self.sentiment_col]

            return tokens, emotion, sentiment

        text = [[torch.as_tensor(self.vocab.label_encoder.encode(str(word))) for word in sentence.words] for sentence in self.vocab.document[idx]]

class LMDataset(Dataset):
    def __init__(self, vocab, freeze_embeddings=True):
        self.vocab = vocab
        self.frozen_embeddings=freeze_embeddings

    def __len__(self):
        return len(self.vocab.document)

    def __getitem__(self, idx):
        words = self.vocab.document[idx].words
        
        if not self.frozen_embeddings:            
            tokens = self.vocab[idx]

            return [np.array(tok) for tok in tokens[:-1]] if len(tokens)>1 else [np.array(tokens[0])], torch.as_tensor(self.vocab.label_encoder.encode(str(words[-1])))

        return [self.vocab.label_encoder.encode(str(word)) for word in words[:-1]] if len(words)>1 else [self.vocab.label_encoder.encode(str(word)) for word in words], torch.as_tensor(self.vocab.label_encoder.encode(str(words[-1]), enc_unk=False))

class WordEmbeddingDataset(Dataset):
    def __init__(self, vocab=None, text=None, text_path=None, morphs=None, tokens=None):
        if vocab is None:
            assert text or text_path is not None, "please specify vocab or text for embedding"
            self.vocab = Vocab(document=text, document_path=text_path, morpheme_template=morphs)
        else:
            self.vocab = vocab

        if tokens is not None:
            self.tokens = self.load_tokens(tokens)
        else:
            self.tokens = [np.array([self.vocab.token_encoder.encode(t, enc_unk=False) for t in tok]) for tok in tqdm(self.vocab.tokens)]

    def __len__(self):
        return len(self.vocab.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    def store_tokens(self, path):
        np.save(path, np.array(self.tokens, dtype=object), allow_pickle=True)

    def load_tokens(self, path):
        return np.load(path, allow_pickle=True)

# class LMDataset(Dataset):
#     def __init__(self, tokens, text, context_size):
#         self.text = text
#         self.tokens = tokens
#         self.context_size = context_size

#     def __len__(self):

#         return len(self.tokens)-self.context_size

#     def __getitem__(self, idx):

#         return [self.tokens[idx:idx+self.context_size]][0], self.text[idx+self.context_size]