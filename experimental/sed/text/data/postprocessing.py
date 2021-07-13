import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from torch.autograd import Variable
from tqdm import tqdm

class AttractPreserve(nn.Module):
    def __init__(self):
        super(AttractPreserve, self).__init__()

        self.relu = nn.ReLU()
        self.l2_loss = nn.MSELoss
        self.cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
    
    def attract(self, x, vocab, margin=0.6):

        sim = 1 - torch.sum(x@vocab.T, dim=1)
        sorted_sim, idx = sim.sort()
        
        pos = sorted_sim[-3:]
        neg = sorted_sim[:3]

        return torch.sum(torch.stack([self.relu(margin+n-p) for n in neg for p in pos]), dim=0)

    def preserve(self, x_hat, x, reg_constant=1e-9):
        arr = x_hat-x

        return reg_constant*(torch.sum(LA.norm(arr, ord=2, dim=1)))

    def forward(self, x_hat, vocab):
        x = x_hat
        x_hat /= x_hat.norm(dim=1, keepdim=True)
        vocab /= vocab.norm(dim=1, keepdim=True)

        att_cost = Variable(self.attract(x_hat, vocab), requires_grad=True)
        pres_cost = Variable(self.preserve(x,x_hat), requires_grad=True)

        return att_cost+pres_cost