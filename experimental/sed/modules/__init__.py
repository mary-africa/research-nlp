
import numpy as np
import torch
import torch.nn  as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence

from pathlib import Path

from ..nn import EmbeddingLayer
from ..utils import morpheme_pad_collate

from typing import Iterable, List


# think of moving this outside
class CompositionalEmbeddings(nn.Module):
    def __init__(self, 
                 morpheme_vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 use_cuda: bool = True,
                 **emb_kwargs
                ):
        super(CompositionalEmbeddings, self).__init__()
        self.compose_embeddings = EmbeddingLayer(morpheme_vocab_size, embedding_dim, hidden_dim, **emb_kwargs)        
        self.embedding_dim = embedding_dim
        
        # device to train
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        
    def embed(self, t_morphs_of_word: torch.Tensor):
        return self.compose_embeddings(t_morphs_of_word, [None])
        
    def forward(self, t_morphs_of_word: torch.Tensor):
        return self.embed(t_morphs_of_word)

    def update_weight_from_path(self, sed_embeddings_pth: str):
        self.compose_embeddings.load_state_dict(torch.load(Path(sed_embeddings_pth), map_location=self.device))

    def embed_morphemes(self, morphemes_word_list: Iterable[List[int]]) -> torch.Tensor:
        tok, tok_len = morphemes_word_list
       
        return pad_packed_sequence(
            self.compose_embeddings.check_batched(
                (tok, tok_len)
            ), batch_first=True)[0].squeeze().clone().detach().cpu().numpy()
    
    def fit(self, dataset, batch_size: int = 256):
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


class CompositionalLanguageModel(nn.Module):
    def __init__(self, 
                morphology_embedder: CompositionalEmbeddings, 
                word_count: int, 
                rnn_dim, 
                use_cuda: bool = True,):
        super(CompositionalLanguageModel, self).__init__()
        
        self.morph_embedder = morphology_embedder
        self.embedding_dim = self.morph_embedder.embedding_dim

        self.relu = nn.ReLU()
        self.birnn = nn.GRU(input_size=self.embedding_dim, hidden_size=rnn_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        self.linear = nn.Sequential(
            nn.Linear(rnn_dim*2, self.embedding_dim), 
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(self.embedding_dim, word_count)
        
        # device to train
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

    def forward(self, tensor_words):
        emb_in = self.morph_embedder(tensor_words).unsqueeze(dim=0).unsqueeze(dim=0)
        output, _ = self.birnn(emb_in)
        output = self.linear(torch.mean(output, 1))
        return self.classifier(output)[0]

    def predict_proba(self, emb_in):
        output = self.forward(emb_in)
        return F.softmax(output, dim=1)
    
    def predict(self, emb_in):
        out = self.predict_proba(emb_in)
        return torch.argmax(out, dim=1)

    def update_weight_from_path(self, lang_model_pth: str):
        self.load_state_dict(torch.load(Path(lang_model_pth), map_location=self.device))
