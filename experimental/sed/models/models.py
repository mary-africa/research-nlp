
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils.rnn import pad_packed_sequence

from text.networks.highway import Highway
from text.networks.embeddings import Embeddings
from text.networks.embeddings import WordEmbeddings

from text.data.vocab import Vocab
from text.data.datasets import WordEmbeddingDataset 
from text.utils.config import EmbeddingsConfig, ModelsConfig

def prepare_model(task:str, training=False):
    task = str(task)
    
    if "sent" in task:
        pass
        #model = SentimentAnalysisModel()

    elif "lang" in task:
        
        train_vocab = Vocab(document_path=ModelsConfig.lm_train_text, morpheme_template=ModelsConfig.morphs, token_store_path=ModelsConfig.lm_token_path, sentence_pieces=[3,4,5])
        model = LanguageModel(
                vocab=train_vocab,
                rnn_dim = 512,
                embed_path=ModelsConfig.lm_embed_path,
                weight_path=ModelsConfig.lm_weight_path
                        ).to(ModelsConfig.device)

        if not training:
            model.load_state_dict(torch.load(Path(ModelsConfig.lm_model_path), map_location=torch.device('cpu')))
            model.eval()
    
            return model, train_vocab
                            
        val_vocab = Vocab(document_path=ModelsConfig.lm_val_text, morpheme_template=ModelsConfig.morphs, label_encoder=train_vocab.label_encoder, sentence_pieces=[3,4,5])
        
        return model, train_vocab, val_vocab

    elif "embed" in task:

        if training:
            train_vocab = Vocab(document_path=ModelsConfig.lm_train_text, morpheme_template=ModelsConfig.morphs)#, token_store_path=ModelsConfig.lm_token_path)
            text_data = WordEmbeddingDataset(train_vocab)#, tokens=ModelsConfig.lm_enc_tok_path)

            train_vocab.save_to_file(ModelsConfig.lm_token_path)
            text_data.store_tokens(ModelsConfig.lm_enc_tok_path)

            model = WordEmbeddings(text_data)
            model.save_embeddings(ModelsConfig.lm_weight_path, ModelsConfig.lm_embed_path)

        else:
            train_vocab = Vocab(document_path=ModelsConfig.lm_train_text, morpheme_template=ModelsConfig.morphs, token_store_path=ModelsConfig.lm_token_path)
            # text_data = WordEmbeddingDataset(train_vocab, tokens=ModelsConfig.lm_enc_tok_path)

            model = WordEmbeddings(train_vocab, embed_path=ModelsConfig.lm_embed_path, weight_path=ModelsConfig.lm_weight_path)
                
        return model

    else:
        # return train_vocab
        # assert task is not None, 'please specify model to prepare'
        pass

class SentimentAnalysisModel(Embeddings):
    def __init__(self, num_emotions, num_sentiments, rnn_dim, pretrained=False, in_c=None, token_vocab_size=None, token_store_path=None):
        super(SentimentAnalysisModel, self).__init__(token_vocab_size=token_vocab_size, in_c=in_c, token_store_path=token_store_path)
        
        self.pretrained = pretrained
        self.embedding_dim = EmbeddingsConfig.embedder_params['embedding_dim']

        self.birnn = nn.GRU(input_size=self.embedding_dim, hidden_size=rnn_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim), 
            nn.ReLU(),
            nn.Dropout(0.4)
            )

        self.classifier1 = nn.Linear(256, num_emotions)
        self.classifier2 = nn.Linear(256, num_sentiments)
      
    def forward(self, inputs):
        
        if not self.pretrained:
            emb_in = self.get_embeddings(inputs, return_array=False)
        else:
            emb_in = torch.stack(inputs[0])
    
        output,_ = self.birnn(emb_in) if self.compose_embeddings.comp_fn is None else pad_packed_sequence(self.birnn(emb_in), batch_first=True)
        output = self.linear(output)  
        
        return torch.mean(self.classifier1(output), 1), torch.mean(self.classifier2(output), 1)


class LanguageModel(Embeddings):
    def __init__(self, vocab, rnn_dim, embed_path=None, weight_path=None):
        super(LanguageModel, self).__init__(vocab=vocab, embed_path=embed_path, weight_path=weight_path)
        
        self.out_c = EmbeddingsConfig.embedder_params['out_c']
        self.embedding_dim = EmbeddingsConfig.embedder_params['embedding_dim']

        self.relu = nn.ReLU()
        self.birnn = nn.GRU(input_size=self.embedding_dim, hidden_size=rnn_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        self.linear = nn.Sequential(
            nn.Linear(rnn_dim*2, self.embedding_dim), 
            nn.ReLU(),
            nn.Dropout(0.2),
            )

        self.classifier = nn.Linear(self.embedding_dim, len(vocab.glossary)+1)


    def forward(self, inputs):
        emb_in = self.get_embeddings(inputs)
        # print(pad_packed_sequence(emb_in).shape)

        output,_ = self.birnn(emb_in) if self.compose_embeddings.comp_fn is None else pad_packed_sequence(self.birnn(emb_in)[0], batch_first=True)
        output = self.linear(torch.mean(output, 1))  
        
        return self.classifier(output)

        # emb_in = self.get_embeddings(inputs, return_array=False)
        # output = self.linear(emb_in)
        # return torch.mean(self.classifier(output), 1)

    # def get_next_words(self, string):

    #         self.check_embeddings() 
    #         token = self.tokenizer.transform(string)[0]
    #         val = self.get_word_embedding(string)
            
    #         sim_dict = {}
    #         sims = [(txt, cosine_similarity(val.reshape(1,-1), vec_.reshape(1,-1)).reshape(1)[0]) for txt,vec in self.embeddings.items() for vec_ in vec if txt!=string or not (vec_==val).all()]

    #         sim_dict[string] = sims
    #         most_similar = self.get_most_similar(string, sim_dict)

    #         return most_similar[0]

    # def predict_next(self, text, predict_size=1):
    #     if predict_size == 1:            
    #         return self.get_next_words(text)

    #     for i in range(predict_size-1):
    #         text = text+list(self.get_next_words(text).values())[0]

    #     return list(self.get_next_words(text).values())[0]


        # for emb, string in zip(emb_in, strings):
        #     if string in self.embeddings:
        #         self.embeddings[string].append(emb)

        #     else:
        #         self.embeddings[string] = [emb]
            
