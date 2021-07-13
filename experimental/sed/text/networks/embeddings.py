# -*- coding: utf-8 -*-

import torch
import copy
import json
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from time import time

from text.data.datasets import lm_pad_collate
from text.data.vocab import retrieve_tokens

from text.networks.highway import Highway
from text.networks.attention import Encoder, EncoderLayer
from text.networks.attention import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding
from text.utils.config import EmbeddingsConfig, ModelsConfig, DataConfig

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, return_array, num_attn_layers=None, d_ff=None, hidden=None, out_c=None, token_vocab_size=None,
                    kernel=None, embed_path=None, max_len=None, composition_fn='non'):
        super(EmbeddingLayer, self).__init__()

        self.comp_fn = composition_fn
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.return_array = return_array
        
        self.emb_mod = nn.Embedding(token_vocab_size, embedding_dim, padding_idx=0)
        self.select_comp(dropout, num_attn_layers, d_ff, hidden, out_c, kernel)

    def select_comp(self, dropout, num_attn_layers, d_ff, hidden, out_c, kernel):
        """Helper to select composition function"""
        if self.comp_fn == 'rnn':
            self.compose = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
            self.comp_linear = nn.Sequential(
                nn.Linear(self.hidden_dim*2, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
                )

        elif self.comp_fn == 'attn':
            c = copy.deepcopy

            attn = MultiHeadedAttention(hidden, self.hidden_dim)
            ff = PositionwiseFeedForward(self.hidden_dim, d_ff, dropout)
            position = PositionalEncoding(self.hidden_dim, dropout)
        
            self.attn = Encoder(EncoderLayer(hidden_dim, c(attn), c(ff), dropout), num_attn_layers)

            self.compose = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
            self.comp_linear = nn.Sequential(
                nn.Linear(self.hidden_dim*2, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
                )
                
        elif self.comp_fn == 'cnn':

            self.cnn = nn.ModuleList([nn.Conv1d(in_channels=self.max_len, out_channels=out_c, kernel_size=k) for k in range(1,kernel+1)])
            self.rnn = nn.GRU(input_size=out_c, hidden_size=self.embedding_dim, num_layers=1, batch_first=True, bidirectional=True)
            self.highway = Highway(self.out_c, 3, f=torch.nn.functional.relu)

            self.linear = nn.Sequential(
                nn.Linear(self.embedding_dim*2, 256), 
                nn.ReLU(),
                nn.Dropout(0.4)
                )
        else:
            self.compose = None


    def padded_sum(self, data, input_len = None, dim=0):
        """
        summing over padded sequence data
        Args:
            data: of dim (batch, seq_len, hidden_size)
            input_lens: Optional long tensor of dim (batch,) that represents the
                original lengths without padding. Tokens past these lengths will not
                be included in the sum.

        Returns:
            Tensor (batch, hidden_size)

        """
        if input_len is not None:
            return torch.stack([
                torch.sum(data[:, :input_len, :], dim=dim)
            ])
        else:
            return torch.stack([torch.sum(data, dim=dim)])

    def rnn_compose(self, emb_in):
        """
        RNN composition of morpheme vectors into word embeddings
        """
                           
        return self.compose(emb_in)[0]
        
    def attn_compose(self, emb_in):
        """
        Composition of morpheme vectors into word embeddings using attention model
        """      
        return self.rnn_compose([self.attn(emb_in[0], None)], None)

    def cnn_compose(self, emb_in):
        """
        CNN composition of morphemes into word embeddings
        """
        assert self.max_len is not None, 'please specify the maximum length of words in vocabulary'
        
        return torch.cat([torch.max(torch.tanh(self.conv(emb_in)), 2)[0] for self.conv in self.cnn])

    def get_composition(self, emb_in, in_len, dim):
        """
        Helper function to get word embeddings from morpheme vectors. Uses additive function by default
        if composition function is not specified
        """
        if self.comp_fn is not None:
            if self.comp_fn == 'rnn':
                if len(in_len)>1 or (len(in_len)==1 and in_len[0] is not None):
                    
                    emb_in = pack_padded_sequence(emb_in, in_len, batch_first=True, enforce_sorted=False)
                    rnn_out,_ = pad_packed_sequence(self.rnn_compose(emb_in), batch_first=True)

                else:
                    rnn_out = self.rnn_compose(emb_in)[0].unsqueeze(0)

                # if self.return_array:
                    
                #     return torch.mean(self.comp_linear(rnn_out),1).clone().detach().cpu().numpy()
                
                return torch.mean(self.comp_linear(rnn_out),1).squeeze(0).clone().detach().cpu()

            elif self.comp_fn == 'attn':
                return self.attn_compose(emb_in)

            elif self.comp_fn == 'cnn':
                # emb_in = [torch.stack(emb_in)]
                if self.return_array:
                    return self.cnn_compose(emb_in).clone().detach().cpu().numpy()
                
                out = self.cnn_compose(emb_in)
                out = self.highway(out)
                
                output,_ = self.birnn(out)
                return self.linear(output)  

        return self.padded_sum(emb_in, in_len, dim=dim)

    def check_input(self, x_in, in_len):
        """
        Helper function to ensure inputs are tensors and in the appropriate device
        """
        if not all(torch.is_tensor(x) for x in x_in):
            x_in = [torch.as_tensor(x).to(EmbeddingsConfig.device).long() for x in x_in]

        if not all(torch.is_tensor(il) for il in in_len):
            in_len = [torch.as_tensor(il).to(EmbeddingsConfig.device).long().reshape(1,) if il is not None else il for il in in_len]
       
        return x_in, in_len

    def forward(self, x_in, in_len, dim=1):
        """
        Get embeddings from morpheme vectors after passing label-encoded vectors through the embedding layer
        Args:
            x_in   - label-encoded vector inputs
            in_len - original lengths of vectors if were padded, else is None

        Returns:
            vector representation (embeddings) of the text sequence 
        """  
        x_in, in_len = self.check_input(x_in, in_len)
        emb_in = torch.cat([torch.stack([self.emb_mod(x)]) for x in x_in])
        
        return self.get_composition(emb_in, in_len, dim)

class Embeddings(nn.Module):
    def __init__(self, vocab, weight_path=None, embed_path=None, frozen=EmbeddingsConfig.freeze_embeddings, return_array=False):
        super(Embeddings, self).__init__()

        self.vocab = vocab
        self.frozen = frozen
        self.embeddings = None
        self.compose_embeddings = EmbeddingLayer(token_vocab_size=len(vocab)+1, max_len=len(max(vocab.token_encoder.items, key=len)), return_array=return_array, **EmbeddingsConfig.embedder_params).to(EmbeddingsConfig.device)

        if embed_path is not None:
            self.load_embeddings(weight_path, embed_path)

        if self.frozen:
            assert self.embeddings is not None, "please specify embeddings dict"

        # Initialize parameters with Glorot / fan_avg.
        if EmbeddingsConfig.embedder_params['composition_fn'] == 'attn':
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
                              else self.get_word_embedding(self.vocab.label_encoder.decode(enc_word), return_array=False)\
                                   for enc_word in words[:_len]
                                    ]) for words,_len in zip(*inputs)]

            emb_len = [torch.as_tensor(len(emb)).long().reshape(1,) for emb in emb_in]
            # print([emb.shape for emb in emb_in],pad_sequence(emb_in, batch_first=True).shape)
            pad_in = pad_sequence(emb_in, batch_first=True)#.to(EmbeddingsConfig.device)
            packed_out = pack_padded_sequence(pad_in, emb_len, batch_first=True, enforce_sorted=False).to(EmbeddingsConfig.device)
            
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
        np.save(embed_store_path, self.embeddings, allow_pickle=True)
   
    def load_embeddings(self, weight_store_path, embed_store_path=None):
        '''
        load embeddings from path on disk
        '''
        self.compose_embeddings.load_state_dict(torch.load(Path(weight_store_path)))
        self.compose_embeddings.eval()

        if embed_store_path is not None:
            self.embeddings = np.load(embed_store_path, allow_pickle=True)

    def get_word_embedding(self, string, return_array=True):
        """
        get word embedding of given string. 

        Args:
            string - string whose vector representation is to be obtained

        Returns:
            Embedding vector of given string
        """
        tokens = torch.stack([torch.as_tensor([self.vocab.token_encoder.encode(tok, enc_unk=False) for tok in retrieve_tokens(self.vocab.morpheme_encoder, [string])]).long()]).to(EmbeddingsConfig.device)
        
        if return_array:
            return self.compose_embeddings(tokens, [None], 1).clone().detach().cpu().numpy()

        return self.compose_embeddings(tokens, [None], 1)

class WordEmbeddings(Embeddings):
    def __init__(self, data, batch_size=256, embed_path=None, weight_path=None):
        super(WordEmbeddings, self).__init__(vocab=data.vocab, return_array=False, frozen=False, embed_path=embed_path, weight_path=weight_path)

        self.data = data
        self.batch_size = int(batch_size)
        self.check_embeddings()
        
    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, key):
        pass

    def embed_string(self,tokens:list):  
        
        str_enc = torch.as_tensor([0])
        
        tok, tok_len, str_enc = lm_pad_collate([(tokens, str_enc[0])], lang_mod=False, frozen_embeddings=False)
        
        tok = [t.to(EmbeddingsConfig.device) for t_ in tok for t in t_.transpose(0,1)]
       
        return pad_packed_sequence(self.check_batched((torch.stack(tok), tok_len)), batch_first=True)[0].squeeze().clone().detach().cpu().numpy()

    def embed(self, weight_store_path=None, emb_store_path=None):
        """
        get embeddings and store dict of each word from corpus and its corresponding embedding vector

        Args:
            tokens     - collection of label-encoded inputs
            text       - original text corresponding to their encodings
            text_path  - path to original text
            store_path - location on disk to save the embeddings

        """

        # _tok_emb = [tok.reshape(1,-1) for tokens in [self.embed_string(*self.data[i*self.batch_size:(i+1)*self.batch_size]) for i in tqdm(range(int((len(self.data)//self.batch_size)+1)))] for tok in tokens]
        # emb_dict = {string : token for string,token in tqdm(zip(self.data.vocab.label_encoder.encode(self.data.vocab.labels), _tok_emb))}
        _tok_emb = [self.embed_string(self.data[i*self.batch_size:(i+1)*self.batch_size]) for i in tqdm(range(int((len(self.data)//self.batch_size)+1)))]
        embeddings = np.concatenate(_tok_emb)

        # self.embeddings.update(emb_dict)
        self.embeddings = embeddings
        self.compose_embeddings.eval()

        
        if emb_store_path is not None or weight_store_path is not None:
            self.save_embeddings(weight_store_path, emb_store_path)
            # self.tokenizer.save_to_disk(text, './str.txt')

    def check_embeddings(self):
        """
        check for existing embeddings and if none are found checks whether embeddings were passed in as an agument
        """
        if self.embeddings is None:
            self.embed()
        
        assert self.embeddings is not None, 'no embeddings found'       

    def get_most_similar(self, string, sim_dict, threshold):
        """
        get most similar word(s) from collection of related words using the cosine similarity measure

        Args:
            string    - string whose most siilar words are to be obtained
            sim_dict  - dictionary of similar words
            threshold - minimum cosine similarity value for words to be considered most similar to string
                        if None then only word with highest cosine similarity is returned
        
        Returns:
            collection of most similar words as determined by their cosine similarity to the string being considered
        """

        cos_sim = [sim[1] for sim in sim_dict[string]]
        max_sim = max(cos_sim)

        if threshold is not None:
            assert max_sim>=threshold, 'threshold set too high, no similar words found'

            return [v for v in sim_dict[string] if v[1]>=threshold]

        return [v for v in sim_dict[string] if v[1]==max_sim]

    def get_similar_words(self, string, k_dim=0, threshold=None):
        """
        get collection of closely related words usnig the cosine similarity of their embedding vectors

        Args:
            string          - string whose related words are to be obtained
            embeddings_dict - dictionary of word embeddings. If embedder already trained uses existing embeddings.
            threshold       - minimum cosine similarity for word to be considered similar to given word

        Returns:
            dictionary of similar words and their similarity as measure by the cosine similarity between their embedding vectors
            and that of the string
        """
        self.check_embeddings()
        val = self.get_word_embedding(string)[k_dim,:]
        
        sim_dict = {}
        sim_dict[string] = [(txt, cosine_similarity(val.reshape(1,-1), vec.reshape(1,-1)).reshape(1)[0]) for txt,vec in enumerate(self.embeddings) if txt!=string or not (vec==val).all()]
        
        most_similar = self.get_most_similar(string, sim_dict, threshold)
        sim_dict[string] = sorted(most_similar,key=lambda x: x[1], reverse=True)

        return sim_dict

    def get_best_analogy(self, sim_list, string_b, return_cos_similarity):
        """
        get most relevant analogy from collection of analogous words. uses cosine similarity measure to determine 
        the best analogy

        Args:
            sim_list - list of words similar to the given word
            string_b - word whose analogy is to be determined
            return_cosine_similarity - whether or not output should include the analogy's cosine similarity

        Returns:
            analogy of the given word
        """
        sorted_sim = sorted([sim for sim in sim_list if sim[1]>0], key=lambda x:x[1], reverse=True)
        max_sim = sorted([sim for sim in sim_list if sim[1]>0], key=lambda x:x[1], reverse=True)[0][0]
        
        if not return_cos_similarity:
            sorted_sim = [sim[0] for sim in sorted_sim]

        # temp = sorted([sim for sim in sim_list if sim[1]>0], key=lambda x:x[1], reverse=True)[0:3]
        # print([self.vocab.label_encoder.decode(tem[0]) for tem in temp], temp)

        if max_sim == self.vocab.label_encoder.encode(string_b, enc_unk=False):
            return self.vocab.label_encoder.decode(sorted_sim[1])
        
        return self.vocab.label_encoder.decode(sorted_sim[0])

    def _3_cos_add(self, a, _a, b, string_b, k_dim, return_cos_similarity):
        """
        determine the analogy of the given word based on an additive function of cosine similarities

        Args:
            a,_a     - vector representation of the example of a word and its corresponding analogy
            b        - vecor representation of the string whose analogy is to be determined
            string_b - string whose analogy is to be determined

        Returns:
            analogy of the string based on given example and determined using cosine similarity
        """
        _b = b - a + _a

        sim_list = [(txt, cosine_similarity(vec.reshape(1,-1),_b).reshape(1)[0]) for txt,vec in enumerate(self.embeddings)]
  
        return self.get_best_analogy(sim_list, string_b, return_cos_similarity)

    def _3_cos_mul(self, a, _a, b, string_b, k_dim, return_cos_similarity, eps=0.001):
        """
        determine the analogy of the given word based on a multiplicative function of cosine similarities

        Args:
            a,_a     - vector representation of the example of a word and its corresponding analogy
            b        - vecor representation of the string whose analogy is to be determined
            string_b - string whose analogy is to be determined

        Returns:
            analogy of the string based on given example and determined using cosine similarity
        """
        
        sim_list = [(txt, (cosine_similarity(vec.reshape(1,-1),b).reshape(1)[0]*cosine_similarity(vec.reshape(1,-1),_a).reshape(1)[0])/(cosine_similarity(vec.reshape(1,-1),a).reshape(1)[0]+eps))\
                    for txt,vec in enumerate(self.embeddings)]
        return self.get_best_analogy(sim_list, string_b, return_cos_similarity)

    def pair_direction(self, a, _a, b, string_b, k_dim, return_cos_similarity):
        """
        determine the analogy of the given word based on an additive function of cosine similarities that maintains
        the ...

        Args:
            a,_a     - vector representation of the example of a word and its corresponding analogy
            b        - vecor representation of the string whose analogy is to be determined
            string_b - string whose analogy is to be determined

        Returns:
            analogy of given string based on given example and determined using cosine similarity
        """
        _b = _a - a

        sim_list = [(txt, cosine_similarity(vec.reshape(1,-1)-b,_b).reshape(1)[0]) for txt,vec in enumerate(self.embeddings)]

        return self.get_best_analogy(sim_list, string_b, return_cos_similarity)

    def get_analogy(self, string_a, analogy_a, string_b, k_dim=0, return_cos_similarity=False):
        """
        get analogous words using 3COSADD, PAIRDIRECTION, or 3COSMUL which make use of the cosine similarity of the embedding vectors.        
        adapted from: https://www.aclweb.org/anthology/W14-1618

        Args:
            string_a, analogy_a - example of a string and its analogy
            string_b - string whose analogy is to be determined
            embeddings_dict - dictionary of embeddings. uses existing embeddings if was pretrained
            return_cosine_similarity - whether or not output should include the analogy's cosine similarity
        
        Returns:
            analogy of given string based on given example and determined using cosine similarity
        """
        self.check_embeddings()
        a, _a, b = (self.get_word_embedding(string)[k_dim,:].reshape(1,-1) for string in [string_a, analogy_a, string_b])
        # print(a,_a,b)
        
        if self.compose_embeddings.comp_fn is None:
            return self._3_cos_add(a, _a, b, string_b, k_dim, return_cos_similarity)
            
        return self._3_cos_mul(a, _a, b, string_b, k_dim, return_cos_similarity) 
