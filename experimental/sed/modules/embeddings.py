from typing import Iterable, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import DataLoader

from ..nn import SEDWordEmbeddingLayer
from ..utils import morpheme_pad_collate, MorphemeDataset

class SEDWordEmbeddings(object):
    def __init__(self, 
                 morpheme_vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 use_cuda: bool = True
                ):
        self.swe_layer = SEDWordEmbeddingLayer(morpheme_vocab_size, embedding_dim, hidden_dim)
        self.swe_layer.eval() # freeze
        
        self.embedding_dim = embedding_dim
        
        # device to train
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        
    def embed(self, t_morphs_of_word: torch.Tensor):
        return self.swe_layer(t_morphs_of_word, [None])

    def embed_morphemes(self, morphemes_word_list: Iterable[List[int]]) -> torch.Tensor: 
#         str_enc = torch.as_tensor([0])
        
#         tok, tok_len, str_enc = lm_pad_collate([(morpheme_indices, str_enc[0])])
#         tok, tok_len = lm_pad_collate([morphemes_word_list])
        
        tok, tok_len = morphemes_word_list
       
        return pad_packed_sequence(
            self.swe_layer.check_batched(
                (tok, tok_len)
            ), batch_first=True)[0].squeeze().clone().detach().cpu().numpy()
    
    def fit(self, dataset: MorphemeDataset, batch_size: int = 256):
        dl = DataLoader(dataset, batch_size=batch_size, collate_fn=morpheme_pad_collate)

        try:
            # Checks if there if there is tqdm 
            from tqdm import tqdm
            _tok_emb = [self.embed_morphemes(d) for d in tqdm(dl)]
        except ImportError:
            _tok_emb = [self.embed_morphemes(d) for d in dl]

        # returns the embeddings
        embeddings = np.concatenate(_tok_emb)
        return embeddings
