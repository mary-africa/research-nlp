import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from typing import Tuple, List, Union


class SEDWordEmbeddingLayer(nn.Module):
    def __init__(self, 
                 token_vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int, 
                 dropout: int = 0.4, 
                 num_attn_layers=None, 
                 d_ff=None, 
                 hidden=None, 
                 out_c=None,
                 kernel=None,
                 padding_idx: int = 0,
                 composition_fn='rnn'):
        super(SEDWordEmbeddingLayer, self).__init__()
        
        self.comp_fn = composition_fn
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.emb_mod = nn.Embedding(token_vocab_size, embedding_dim, padding_idx=padding_idx)
        self.compose, self.comp_linear = self.select_comp(dropout, num_attn_layers, d_ff, hidden, out_c, kernel)
    
    def select_comp(self, dropout, num_attn_layers, d_ff, hidden, out_c, kernel):
        """Helper to select composition function"""
        compose = None
        comp_linear = None
        
        if self.comp_fn == 'rnn':
            compose = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
            comp_linear = nn.Sequential(
                nn.Linear(self.hidden_dim*2, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
        return compose, comp_linear

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
    
    def get_composition(self, emb_in, in_len, dim):
        """
        Helper function to get word embeddings from morpheme vectors. Uses additive function by default
        if composition function is not specified
        """
        if self.comp_fn == 'rnn':
            if len(in_len)>1 or (len(in_len)==1 and in_len[0] is not None):

                emb_in = pack_padded_sequence(emb_in, in_len, batch_first=True, enforce_sorted=False)
                rnn_out,_ = pad_packed_sequence(self.rnn_compose(emb_in), batch_first=True)

            else:
                rnn_out = self.rnn_compose(emb_in)[0].unsqueeze(0)

            # if self.return_array:

            #     return torch.mean(self.comp_linear(rnn_out),1).clone().detach().cpu().numpy()

            return torch.mean(self.comp_linear(rnn_out),1).squeeze(0).clone().detach().cpu()

        return self.padded_sum(emb_in, in_len, dim=dim)
    
    
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
        emb_in = [self.forward(x_in, in_len) for x_in, in_len in zip(inputs[0], inputs[1])]

        if self.comp_fn is not None:
            emb_len = [torch.as_tensor(len(emb)) for emb in emb_in]
            pad_in = pad_sequence(emb_in, batch_first=True)
            packed_out = pack_padded_sequence(pad_in, emb_len, batch_first=True, enforce_sorted=False)
            
            return packed_out
        
        return torch.cat(emb_in, dim=0)

    def forward(self, x_in, in_len, dim=1):
        """
        Get embeddings from morpheme vectors after passing label-encoded vectors through the embedding layer
        Args:
            x_in   - label-encoded vector inputs
            in_len - original lengths of vectors if were padded, else is None

        Returns:
            vector representation (embeddings) of the text sequence 
        """  
#         x_in, in_len = self.check_input(x_in, in_len)
        emb_in = torch.cat([torch.stack([self.emb_mod(x)]) for x in x_in])
        
        return self.get_composition(emb_in, in_len, dim)


class SEDSequenceLayer(nn.Module):
    def __init__(self, word_count: int, embedding_dim: int, comp_fn: str = None, rnn_dim: int = 32):
        super(SEDSequenceLayer, self).__init__()
        self.comp_fn = comp_fn
        self.birnn = nn.GRU(
            input_size=embedding_dim, 
            hidden_size=rnn_dim, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Sequential(
            nn.Linear(rnn_dim*2, embedding_dim), 
            nn.ReLU(),
            nn.Dropout(0.2),
            )
        self.classifier = nn.Linear(embedding_dim, word_count)

    def forward(self, emb_in):    
#         output, _ = self.birnn(emb_in) if self.comp_fn is None else pad_packed_sequence(self.birnn(emb_in)[0], batch_first=True)
        output, _ = self.birnn(emb_in)
        output = self.linear(torch.mean(output, 1))  
        
        return self.classifier(output)

    def predict_proba(self, emb_in):
        output = self.forward(emb_in)
        return F.softmax(output, dim=1)
    
    def predict(self, emb_in):
        out = self.predict_proba(emb_in)
        return torch.argmax(out, dim=1) 

class SEDLanguageModel(nn.Module):
    def __init__(self, 
                 embeddings: torch.Tensor,
                 word_count: int, 
                 comp_fn: str = None,
                 rnn_dim: int = 32):
        super(SEDLanguageModel, self).__init__()
        self.embeddings = embeddings
        self.look_up = SEDSequenceLayer(word_count, embeddings.shape[-1], comp_fn, rnn_dim)

    def forward(self, padded_word_sequence: List[Tuple[int]]):
        """
        padded_word_sequence: BATCH FIRST
        """
        out = F.one_hot(padded_word_sequence).to(torch.float) # [batch, #words, #morphmets]
        out = torch.matmul(out, self.embeddings)
        return self.look_up(out)
