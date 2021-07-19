
import torch
from ..data.datasets import pad_collate, lm_pad_collate

class DataConfig():
    dataloader_params = dict(        
    batch_size=64,
    collate_fn= lambda x: pad_collate(x, pretrained=False),
    shuffle=False,
    num_workers=2, 
    # pin_memory=True
    )

class ModelsConfig():

    # setting random seed
    torch.manual_seed(7)

    # checks if gpu is available
    use_cuda = torch.cuda.is_available() 
    device = torch.device("cuda:0" if use_cuda else "cpu")

    learning_rate = 1e-4

    #for the Language Model 
    lang_mod = True
    morphs = './sed/df_morphs.txt'
    lm_train_text = './sed/train.txt'
    lm_val_text = './sed/valid.txt'

    lm_token_path = './sed/tokens/small-tkns.txt'
    lm_enc_tok_path = "./sed/tokens/small-enc-tkns.txt.npy"

    lm_weight_path = './sed/embeddings/small-embeddings.pth'
    lm_embed_path = './sed/embeddings/small-embeddings.txt.npy'
    lm_model_path = './sed/models/6.9453_lm.pth'
    
    # Configurations
    # model_params = dict(
    # embedding_dim = 300,
    # rnn_dim = 256,
    # hidden_dim = 256,
    # composition_fn = 'rnn'
    # )

    EPOCHS = 100

class EmbeddingsConfig():
    # setting random seed
    torch.manual_seed(7)

    # checks if gpu is available
    use_cuda = torch.cuda.is_available() 
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # Configurations
    embedder_params = dict(
    embedding_dim = 512,
    hidden_dim = 512,
    dropout=0.2,
    num_attn_layers=6, 
    d_ff=2048,
    hidden=8,
    composition_fn = 'rnn',
    out_c = 32,
    kernel = 7
    )
    freeze_embeddings=True