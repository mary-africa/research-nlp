 
import torch 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import re
import string
import numpy as np
from time import time
from pathlib import Path
from tqdm.notebook import tqdm
from multiprocessing import Pool

from text.data.structures import Document
from text.data.postprocessing import AttractPreserve
from text.data.datasets import lm_pad_collate, WordEmbeddingDataset

from text.networks.embeddings import WordEmbeddings
from text.utils.models import prepare_model
from text.utils.config import DataConfig, ModelsConfig, EmbeddingsConfig

class Inference():
    def __init__(self, task=None):
        if 'model' in task:
            self.model, self.train_vocab = prepare_model(task=task)

        self.word_embeddings = prepare_model(task='embeddings')

    def tokenize(self, text):
        
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        text = Document(document_text=text, morph_path=ModelsConfig.morphs, sentence_pieces=[1])

        return [sentence[i] for sentence in text for i in range(len(sentence))]

    def word_analogy(self, example_word, example_analogy, word):
        example_word, example_analogy, word = example_word.lower(), example_analogy.lower(), word.lower()
        return self.word_embeddings.get_analogy(example_word, example_analogy, word)

    def get_proba(self, text):

        str_enc = [torch.as_tensor(self.train_vocab.label_encoder.encode(txt)) for txt in text]
        output = self.model(([torch.stack(str_enc)], torch.stack([torch.as_tensor(len(str_enc)).long().reshape(1,)])))
        
        sm = torch.nn.Softmax(dim=1)
    #     print([sm(out).detach().cpu().numpy() for out in [output]])
        return [sm(out).detach().cpu().numpy() for out in [output]]

    def get_preds(self, text):

        if isinstance(text, str):
            text = [txt for txt in text.lower().translate(str.maketrans('', '', string.punctuation)).split(' ') if txt not in '']
                    
        max_proba = np.argmax(self.get_proba(text))
        # print(np.argsort(-self.get_proba(text)[0])[:3], [self.train_vocab.label_encoder.decode(p) for pr in np.argsort(-self.get_proba(text)[0]) for p in pr[:3]])
        return self.train_vocab.label_encoder.decode(max_proba)

        
class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

class Pipeline(object):
    def __init__(self, train_data, test_data, model, save_path=None, resume_from_checkpoint=False, model_weights=None):
        
        self.model = model            
        self.resume_train = resume_from_checkpoint
        self.model_weights = model_weights

        if self.resume_train:
            assert model_weights is not None, "specify model weights save location"
            self.load_model_checkpoint(self.model_weights)

        if ModelsConfig.lang_mod:
            DataConfig.dataloader_params['collate_fn'] = lambda x: lm_pad_collate(
                x,
                lang_mod=True,
                frozen_embeddings=EmbeddingsConfig.freeze_embeddings,
                )
        
        self.train_loader = DataLoader(train_data, **DataConfig.dataloader_params)
        self.test_loader = DataLoader(test_data, **DataConfig.dataloader_params)

        self.criterion = F.cross_entropy
        self.optimizer = optim.AdamW(self.model.parameters(), lr=ModelsConfig.learning_rate)
        
        self.iter_meter = IterMeter()
        self.save_path = save_path
        self.model_checkpoints = []

    def train(self, epoch, verbose):
        self.model.train()
        
        start_time = time()
        data_len = len(self.train_loader.dataset)

        for batch_idx, _data in enumerate(self.train_loader):
            x, x_len, y = _data 

            y = [y_.to(ModelsConfig.device) for y_ in [y]]
            y = [y_.reshape(y_.shape[0],) for y_ in y]

            self.optimizer.zero_grad()

            output = self.model((x, x_len))
            loss = self.criterion(output, y[0])
            loss.backward()

            self.optimizer.step()
            if batch_idx % verbose == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPerplexity: {:5.2f}\telapsed: {:.2f} mins'.format(
                    epoch, batch_idx * len(x), data_len,
                    100. * batch_idx / len(self.train_loader), loss.item(), np.exp(loss.item()), (time()-start_time)/60))#, loss2.item()))\tap_Loss: {:.6f}
                
    def test(self, epoch):
        print('\nevaluating…')
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for I, _data in enumerate(self.test_loader):
                x, x_len, y = _data  

                y = [y_.to(ModelsConfig.device) for y_ in [y]]
                y = [y_.reshape(y_.shape[0],) for y_ in y]

                output = self.model((x, x_len))
                loss = torch.sum(torch.stack([self.criterion(out, y) for out,y in zip([output], y)]))
                test_loss += loss.item() / len(self.test_loader)

        print('Test set: Average loss: {:.4f} | Perplexity: {:8.2f}\n'.format(test_loss, np.exp(test_loss)))

        return round(test_loss, 4), self.model.state_dict()
        
    def train_model(self, early_stop=3, verbose=10):
        
        accum_loss = torch.tensor(float('inf')) if not self.resume_train else torch.tensor(float(re.findall("\d+\.\d+", self.model_weights)[0]))
        stop_eps = 0 
        try:
            for epoch in range(1, ModelsConfig.EPOCHS + 1):
                self.train(epoch, verbose)             
                test_loss, w8 = self.test(epoch)
                
                if test_loss < accum_loss:
                    self.model_checkpoints.append((w8, test_loss))
                    accum_loss = test_loss
                    stop_eps = 0
                else:
                    stop_eps += 1

                if stop_eps >= early_stop:
                    self.save_model_checkpoint()
                    break

        except KeyboardInterrupt:
            self.save_model_checkpoint() if self.model_checkpoints else print("first epoch not completed, model checkpoint will not be saved")

    def save_model_checkpoint(self):
        best_model, accum_loss = self.model_checkpoints[-1]

        if self.save_path is not None and not ModelsConfig.lang_mod:
            torch.save(best_model, Path(self.save_path).joinpath(f'{accum_loss}_SAM.pth'))
        
        if self.save_path is not None and ModelsConfig.lang_mod:
            torch.save(best_model, Path(self.save_path).joinpath(f'{accum_loss}_lm.pth'))

    def load_model_checkpoint(self, model_weights):
        self.model.load_state_dict(torch.load(Path(model_weights)))            

