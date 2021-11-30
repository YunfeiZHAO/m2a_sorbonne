import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import ForwardRef, List

import time
import re
from torch.utils.tensorboard import SummaryWriter





logging.basicConfig(level=logging.INFO)

FILE = "../data/en-fra.txt"

writer = SummaryWriter("runs/tag-"+time.asctime())

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len

def unravel_index(indices, shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=8

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage

class Encoder(nn.Module):

    def __init__(self, dict_size, emb_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(dict_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim)

    def forward(self, batch):
        batch_emb = self.embedding(batch)
        _, hidden = self.gru(batch_emb)
        return hidden

class Decoder(nn.Module):

    def __init__(self, emb_dim, hidden_dim, vocTarget):
        super(Decoder, self).__init__()
        self.dict_size = len(vocTarget)
        self.PAD_id = vocTarget['PAD']
        self.SOS_id = vocTarget['SOS']
        self.EOS_id = vocTarget['EOS']
        self.padding = nn.functional.one_hot(torch.tensor(self.PAD_id), self.dict_size).float()
        self.embedding = nn.Embedding(self.dict_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, self.dict_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, hidden, lenseq, *, k=5):
        batch_size = hidden.shape[1]
        x = torch.ones(k, batch_size).long()*self.SOS_id
        pred = torch.zeros(k, 1, batch_size, self.dict_size)
        res = torch.zeros(k, 0, batch_size, self.dict_size)
        prob = torch.ones(k, batch_size)
        hidden = torch.stack([hidden]*k)
        torch.autograd.set_detect_anomaly(True)
        for i in range(lenseq):
            for j in range(k):
                done = torch.logical_or(x[j, :]  == self.EOS_id, x[j, :] == self.PAD_id)
                not_done = torch.logical_not(done)
                pred[j, :, done, :] = self.padding
                pred[j, :, not_done, :], hidden[j, :, not_done, :] = self.generate(x[j, not_done], hidden[j, :, not_done, :])
            res = torch.cat((res, pred), dim=1)
            for j in range(k):
                pred[j, 0] = pred[j, 0].clone()*prob[j, :, None].clone()
            for j in range(batch_size):
                idx = self.topk_mult(pred[:,0,j,:], k)
                res[:, :, j, :] = res[idx[:,0], :, j, :]
                prob[:, j] = prob[idx[:,0], j]
                hidden[:, :, j, :] = hidden[idx[:,0], :, j, :]
                x[:, j] = idx[:,1]
        for j in range(batch_size):
            idx = self.topk_mult(pred[:,0,j,:], 1)
            res[1, :, j, :] = res[idx[:,0], :, j, :]
        return res[1, :, :, :]

    def teacher_forcing(self, hidden, target, lenseq):
        batch_size = hidden.shape[1]
        x = torch.tensor([self.SOS_id]*batch_size)
        pred = torch.zeros(1, batch_size, self.dict_size)
        res = torch.zeros(0, batch_size, self.dict_size)
        for i in range(lenseq):
            done = torch.logical_or(x  == self.EOS_id, x == self.PAD_id)
            not_done = torch.logical_not(done)
            #if all(done):
            #    break
            pred[:, done, :] = self.padding
            pred[:, not_done, :], hidden[:, not_done, :] = self.generate(x[not_done], hidden[:, not_done, :])
            res = torch.cat((res, pred))
            x = target[i, :]
        return res

    def generate(self, x, hidden):
        x = self.embedding(x)
        x = torch.unsqueeze(x, 0)
        _, hidden = self.gru(x, hidden)
        x = self.linear(hidden)
        x = self.softmax(x)
        return x, hidden

    def topk_mult(self, array, k):
        _, idx = torch.topk(array.flatten(), k)
        return unravel_index(idx, array.shape)


class Translation(nn.Module):

    def __init__(self, emb_dim, hidden_dim, vocSource, vocTarget):
        super(Translation, self).__init__()
        self.encoder = Encoder(len(vocSource), emb_dim, hidden_dim)
        self.decoder = Decoder(emb_dim, hidden_dim, vocTarget)

    def forward(self, batch, lenseq, *, k=5):
        return self.decoder(self.encoder(batch), lenseq, k=k)

    def teacher_forcing(self, batch, target, lenseq):
        return self.decoder.teacher_forcing(self.encoder(batch), target, lenseq)

def accuracy(batch, tags):
    pred = torch.argmax(batch, dim=1)
    return torch.mean((torch.logical_and(pred == tags, tags != 0)).float())

def train(epoch, model, optimizer, train_loader, val_loader, loss_fun, writer):

    iter = epoch * len(train_loader)
    for batch_train, _, tags_train, _ in train_loader:
        iter += 1

        # calculate loss
        #batch_train = introduce_random_OOV(batch_train, id_OOV)
        #pred_train = model(batch_train, tags_train.shape[0])
        pred_train = model.teacher_forcing(batch_train, tags_train, tags_train.shape[0])
        
        pred_train = pred_train.permute((1, 2, 0))
        tags_train = tags_train.T
        loss_train = loss_fun(pred_train, tags_train)
        #print(loss_train)

        # optimization step
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # calculate validation loss
        batch_val, _, tags_val, _ = next(val_loader.__iter__())
        with torch.no_grad():
            pred_val = model(batch_val, tags_val.shape[0])
            pred_val = pred_val.permute((1, 2, 0))
            tags_val = tags_val.T
            loss_val = loss_fun(pred_val, tags_val)

        # log progress
        accuracy_train = accuracy(pred_train, tags_train)
        accuracy_val = accuracy(pred_val, tags_val)
        writer.add_scalar('Loss/train', loss_train, iter)
        writer.add_scalar('Loss/validation', loss_val, iter)
        writer.add_scalar('Accuracy/train', accuracy_train, iter)
        writer.add_scalar('Accuracy/validation', accuracy_val, iter)
        print('[{:2d}/{}] Iteration {:4d}: train loss {:11.9f}, val loss {:11.9f}, train accuracy {:6.3}, val accuracy {:6.3}, learning rate {:6.5f}'.format(
            epoch + 1, 
            cfg["epochs"], 
            iter, 
            loss_train, 
            loss_val, 
            accuracy_train,
            accuracy_val,
            cfg['learning_rate']))

cfg = {'learning_rate': 0.1, 'epochs': 60}
model = Translation(40, 50, vocEng, vocFra)
optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'])
loss_fun = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(cfg['epochs']):
    train(epoch, model, optimizer, train_loader, test_loader, loss_fun, SummaryWriter())


                


