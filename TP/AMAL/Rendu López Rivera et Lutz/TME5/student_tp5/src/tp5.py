
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

#  TODO: 

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    pass


class RNN(nn.Module):
    def __init__(self, inputSize, latentSize, outputSize):
        super(RNN, self).__init__()
        self.inputSize = inputSize
        self.latentSize = latentSize
        self.outputSize = outputSize
        self.linearLatent = nn.Linear((self.inputSize + self.latentSize), self.latentSize)
        self.linearDecode = nn.Linear(self.latentSize, self.outputSize)
        self.funActivation = nn.Tanh()
        self.funDecode = nn.Softmax(dim=-1)

    def one_step(self, x, h):
        return self.funActivation(self.linearLatent(torch.cat((x, h), dim=-1)))

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.shape[1], self.latentSize)
            
        h_ = torch.zeros((x.shape[0], x.shape[1], self.latentSize))
        for i in range(x.shape[0]):
            h = self.one_step(x[i, :, :], h)
            h_[i] = h
        return h_

    def decode(self, h):
        return self.funDecode(self.linearDecode(h))


class LSTM(RNN):
    #  TODO:  Implémenter un LSTM
    pass


class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    pass



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
