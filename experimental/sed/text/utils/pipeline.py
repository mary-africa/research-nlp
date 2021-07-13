 
import torch 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        text = Document(document_text=text, morph_path=DataConfig.morphs)

        return [sentence[i] for sentence in text for i in range(len(sentence))]

    def word_analogy(self, example_word, example_analogy, word):
        
        return self.word_embeddings.get_analogy(example_word, example_analogy, word)

    def get_proba(self, text):
        str_enc = [torch.as_tensor(self.train_vocab.label_encoder.encode(txt)) for txt in text]
        output = self.model(([torch.stack(str_enc)], torch.stack([torch.as_tensor(len(str_enc)).long().reshape(1,)])))
        
        sm = torch.nn.Softmax(dim=1)
    #     print([sm(out).detach().cpu().numpy() for out in [output]])
        return [sm(out).detach().cpu().numpy() for out in [output]]

    def get_preds(self, text, task="language model"):

        if isinstance(text, str):
            text = [txt for txt in text.lower().translate(str.maketrans('', '', string.punctuation)).split(' ') if txt not in '']
                    
        max_proba = np.argmax(self.get_proba(text))
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
    def __init__(self, train_data, test_data, model, save_path=None):
        if ModelsConfig.lang_mod:
            DataConfig.dataloader_params['collate_fn'] = lambda x: lm_pad_collate(
                x,
                lang_mod=True,
                frozen_embeddings=EmbeddingsConfig.freeze_embeddings,
                # word_embedder=self.model.compose_embeddings
                )
        
        self.train_loader = DataLoader(train_data, **DataConfig.dataloader_params)
        self.test_loader = DataLoader(test_data, **DataConfig.dataloader_params)

        self.criterion = F.cross_entropy
        # self.criterion2 = AttractPreserve()
        self.model = model
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

            # if ModelsConfig.lang_mod:
            #     self.criterion = torch.nn.BCEWithLogitsLoss()
            # x, y = [xx.to(ModelsConfig.device) for xx in x], [y_.to(ModelsConfig.device) for y_ in [y]]
            y = [y_.to(ModelsConfig.device) for y_ in [y]]

            # if not ModelsConfig.lang_mod:
            y = [y_.reshape(y_.shape[0],) for y_ in y]

            self.optimizer.zero_grad()

            output = self.model((x, x_len))
            # print([(np.argmax(out.clone().detach().cpu().numpy()), _y) for out,_y in zip(output,y[0])])
            
            loss = self.criterion(output, y[0])#torch.sum(torch.stack([self.criterion(out, y) for out,y in zip([output], y)]))
            # print(loss)
            loss.backward()

            # loss2 = self.criterion2(self.model.classifier.weight.data, torch.stack(self.model.compose_embeddings.mini_vocab).squeeze(1))
            # loss2.backward()

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
                # x, y = [xx.to(ModelsConfig.device) for xx in x], [y_.to(ModelsConfig.device) for y_ in [y]]
                y = [y_.to(ModelsConfig.device) for y_ in [y]]

                # if not ModelsConfig.lang_mod:
                y = [y_.reshape(y_.shape[0],) for y_ in y]

                output = self.model((x, x_len))
                loss = torch.sum(torch.stack([self.criterion(out, y) for out,y in zip([output], y)]))
                # loss = self.criterion(self.model.classifier.weight.data, self.model.compose_embeddings.mini_vocab)
                test_loss += loss.item() / len(self.test_loader)

        print('Test set: Average loss: {:.4f} | Perplexity: {:8.2f}\n'.format(test_loss, np.exp(test_loss)))

        return round(test_loss, 4), self.model.state_dict()
        
    def train_model(self, early_stop=3, verbose=10):
        # print(self.model)
        #train and evaluate model
        accum_loss = torch.tensor(float('inf'))
        stop_eps = 0 

        for epoch in range(1, ModelsConfig.EPOCHS + 1):
            self.train(epoch, verbose)             
            test_loss, w8 = self.test(epoch)
            
            # print(loss2)

            if test_loss < accum_loss:
                self.model_checkpoints.append((w8, test_loss))
                accum_loss = test_loss
                stop_eps = 0
            else:
                stop_eps += 1

            if stop_eps >= early_stop:
                self.save_model()
                break

    def save_model(self):
        best_model, accum_loss = self.model_checkpoints[-1]

        if self.save_path is not None and not ModelsConfig.lang_mod:
            torch.save(best_model, Path(self.save_path).joinpath(f'{accum_loss}_SAM.pth'))
        
        if self.save_path is not None and ModelsConfig.lang_mod:
            torch.save(best_model, Path(self.save_path).joinpath(f'{accum_loss}_lm.pth'))


class LanguageModelPipeline(object):
    def __init__(self, model, train_data, val_data, batch_size=256, save_path=None):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size

        self.criterion = F.cross_entropy
        self.optimizer = optim.AdamW(self.model.parameters(), lr=ModelsConfig.learning_rate)
        
        self.iter_meter = IterMeter()
        self.save_path = save_path
        self.train_phase = True
        self.pool = Pool()
    
    def __len__(self):
        if self.train_phase:
            return (len(self.train_data)//self.batch_size)+1

        return (len(self.val_data)//self.batch_size)+1

    def __getitem__(self, idx):
        if self.train_phase:
            return [self.train_data[i] for i in tqdm(range(idx*self.batch_size,(idx+1)*self.batch_size))]
        
        return [self.val_data[i] for i in range(idx*self.batch_size,(idx+1)*self.batch_size)]


    def train(self, epoch, verbose):
        self.model.train()
        self.train_phase = True
        
        start_time = time()
        data_len = len(self.train_data)
        
        for batch_idx in range(self.__len__()):
            x, x_len, y = lm_pad_collate(self.__getitem__(batch_idx), lang_mod=True, frozen_embeddings=EmbeddingsConfig.freeze_embeddings)
            
            # if ModelsConfig.lang_mod:
            #     self.criterion = torch.nn.BCEWithLogitsLoss()
            x, y = [xx.to(ModelsConfig.device) for xx in x], [y_.to(ModelsConfig.device) for y_ in [y]]

            # if not ModelsConfig.lang_mod:
            y = [y_.reshape(y_.shape[0],) for y_ in y]

            self.optimizer.zero_grad()

            output = self.model((x, x_len))
            
            loss = torch.sum(torch.stack([self.criterion(out, y) for out,y in zip([output], y)]))
            loss.backward()

            # loss2 = self.criterion2(self.model.classifier.weight.data, torch.stack(self.model.compose_embeddings.mini_vocab).squeeze(1))
            # loss2.backward()

            self.optimizer.step()
            if batch_idx % verbose == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPerplexity: {:5.2f}\telapsed: {:.2f} mins'.format(
                    epoch, batch_idx * len(x), data_len,
                    100. * batch_idx / self.__len__(), loss.item(), np.exp(loss.item()), (time()-start_time)/60))#, loss2.item()))\tap_Loss: {:.6f}
                
    def test(self, epoch):
        print('\nevaluating…')
        self.model.eval()
        self.train_phase = False

        test_loss = 0
        with torch.no_grad():
            for ix in range(self.__len__()):
                for data in lm_pad_collate(self.__getitem__(ix), lang_mod=True, frozen_embeddings=EmbeddingsConfig.freeze_embeddings):
                    x, x_len, y = data

                    x, y = [xx.to(ModelsConfig.device) for xx in x], [y_.to(ModelsConfig.device) for y_ in [y]]

                    # if not ModelsConfig.lang_mod:
                    y = [y_.reshape(y_.shape[0],) for y_ in y]

                    output = self.model((x, x_len))
                    loss = torch.sum(torch.stack([self.criterion(out, y) for out,y in zip([output], y)]))
                    # loss = self.criterion(self.model.classifier.weight.data, self.model.compose_embeddings.mini_vocab)
                    test_loss += loss.item() / len(self.test_loader)

        print('Test set: Average loss: {:.4f} | Perplexity {:8.2f}\n'.format(test_loss, np.exp(test_loss)))

        return round(test_loss, 4), self.model.state_dict()
        
    def train_model(self, early_stop=3, verbose=10):
        # print(self.model)
        #train and evaluate model
        accum_loss = torch.tensor(float('inf'))
        stop_eps = 0 
        weights = []
        for epoch in range(1, ModelsConfig.EPOCHS + 1):
            self.train(epoch, verbose)             
            test_loss, w8 = self.test(epoch)
            
            # print(loss2)

            if test_loss < accum_loss:
                weights.append(w8)
                accum_loss = test_loss
                stop_eps = 0
            else:
                stop_eps += 1

            if stop_eps >= early_stop:

                if self.save_path is not None and not ModelsConfig.lang_mod:
                    torch.save(weights[-1], Path(self.save_path).joinpath(f'{accum_loss}_SAM.pth'))
                
                if self.save_path is not None and ModelsConfig.lang_mod:
                    torch.save(weights[-1], Path(self.save_path).joinpath(f'{accum_loss}_lm.pth'))

                break

