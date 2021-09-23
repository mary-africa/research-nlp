from typing import Iterable, List

import numpy as np
import torch
import torch.nn  as nn

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from ..nn import SEDWordEmbeddingLayer, EmbeddingLayer
from ..utils import morpheme_pad_collate, MorphemeDataset

from pathlib import Path


# [OBSELETE]
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


class CompositionalEmbeddings(nn.Module):
    def __init__(self, 
                word_encoder, 
                morpheme_encoder, 
                morphology_analyzer, 
                weight_path=None, 
                embed_path=None, 
                frozen=True, 
                return_array=False, 
                use_cuda=torch.cuda.is_available(),
                **kwargs):
        super(CompositionalEmbeddings, self).__init__()

        self.device=torch.device("cuda:0" if use_cuda else "cpu")

        self.frozen = frozen
        self.embeddings = None
        self.word_encoder = word_encoder
        self.morpheme_encoder = morpheme_encoder
        self.morph_analyzer = morphology_analyzer
        self.compose_embeddings = EmbeddingLayer(
                                        device=self.device, 
                                        token_vocab_size=len(self.morpheme_encoder.items)+1, 
                                        max_len=len(max(self.morpheme_encoder.items, key=len)), 
                                        return_array=return_array, 
                                        **kwargs).to(self.device)

        if self.frozen:
            assert embed_path is not None and weight_path is not None, "please specify word embeddings to use"
            self.load_embeddings(weight_store_path=weight_path, embed_store_path=embed_path)

        # Initialize parameters with Glorot / fan_avg.
        if self.compose_embeddings.comp_fn == 'attn':
            for p in self.compose_embeddings.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def __len__(self):
        return len(self.embeddings)            
    
    def __getitem__(self, index):
        return self.embeddings[index]

    def check_batched(self, inputs):
        """
        check whether data is passed in batches(for models like the language and sentiment analysis)
        or as single inputs and get embeddings accordingly.
        Args:
            inputs - collection containing the label-encoded words as well as their corresponding original
                     lengths if were padded
        Returns:
            tensor of the vector representation of the input sequence
        """
        emb_in = [self.compose_embeddings(x_in, in_len) for x_in, in_len in zip(inputs[0], inputs[1])]

        if self.compose_embeddings.comp_fn is not None:
            emb_len = [torch.as_tensor(len(emb)) for emb in emb_in]
            pad_in = pad_sequence(emb_in, batch_first=True)
            packed_out = pack_padded_sequence(pad_in, emb_len, batch_first=True, enforce_sorted=False)
            
            return packed_out
             
        return torch.cat(emb_in, dim=0)

    def get_embeddings(self, inputs, store_path=None):
        """
        get vector representation of the input data
        Args:
            inputs - collection of the label-encoded words as well as their corresponding original lengths if padded
        Returns:
            vector representation of the input sequence
        """
        if self.frozen:
            assert self.embeddings is not None, 'no embeddings found'
            emb_in = [
                torch.stack([
                    torch.as_tensor(self.__getitem__(enc_word))\
                         if enc_word<self.__len__()\
                              else self.get_word_embedding(self.word_encoder.decode(enc_word), return_array=False)\
                                   for enc_word in words[:_len]
                                    ]) for words,_len in zip(*inputs)]

            emb_len = [torch.as_tensor(len(emb)).long().reshape(1,) for emb in emb_in]
            pad_in = pad_sequence(emb_in, batch_first=True)
            packed_out = pack_padded_sequence(pad_in, emb_len, batch_first=True, enforce_sorted=False).to(self.device)
            
        else:
            packed_out = self.check_batched(inputs)

        if store_path is not None:
            self.save_embeddings(store_path)

        return packed_out

    def save_embeddings(self, weight_store_path, embed_store_path):
        '''
        store embeddings to file on disk
        Args:
            embeddings - dictionary of embedding vectors
            store_path - location on disk to store morphemes
        '''
        torch.save(self.compose_embeddings.state_dict(), Path(weight_store_path))
        torch.save(self.embeddings, embed_store_path)
   
    def load_embeddings(self, weight_store_path=None, embed_store_path=None):
        '''
        load embeddings from path on disk
        '''
        if weight_store_path is not None:
            self.compose_embeddings.load_state_dict(torch.load(Path(weight_store_path), map_location=torch.device('cpu')))
            self.compose_embeddings.eval()

        if embed_store_path is not None:
            self.embeddings = torch.load(embed_store_path).clone().detach().cpu().numpy()

    def get_word_embedding(self, string, return_array=True):
        """
        get word embedding of given string. 
        Args:
            string - string whose vector representation is to be obtained
        Returns:
            Embedding vector of given string
        """
        tokens = torch.stack([torch.as_tensor([self.morpheme_encoder.encode(tok, enc_unk=False) for tok in retrieve_tokens(self.morph_analyzer, [string])]).long()]).to(self.device)
        
        if return_array:
            return self.compose_embeddings(tokens, [None], 1).clone().detach().cpu().numpy()

        return self.compose_embeddings(tokens, [None], 1)

