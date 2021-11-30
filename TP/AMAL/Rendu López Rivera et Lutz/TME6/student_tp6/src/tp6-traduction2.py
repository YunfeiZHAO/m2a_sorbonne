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
from typing import List

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=100

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

    def forward(self, hidden, target, lenseq):
        batch_size = hidden.shape[1]
        x = torch.tensor([self.SOS_id]*batch_size)
        pred = torch.zeros(1, batch_size, self.dict_size)
        res = torch.zeros(0, batch_size, self.dict_size)
        print(lenseq)
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

class Translation(nn.Module):

    def __init__(self, emb_dim, hidden_dim, vocSource, vocTarget):
        super(Translation, self).__init__()
        self.encoder = Encoder(len(vocSource), emb_dim, hidden_dim)
        self.decoder = Decoder(emb_dim, hidden_dim, vocTarget)

    def forward(self, batch, target, lenseq):
        return self.decoder(self.encoder(batch), target, lenseq)

    def teacher_forcing(self, batch, target, lenseq):
        return self.decoder.teacher_forcing(self.encoder(batch), target, lenseq)

def accuracy(batch, tags):
    pred = torch.argmax(batch, dim=1)
    return torch.mean((pred == tags).float())


def train(epoch, model, optimizer, train_loader, val_loader, loss_fun, writer):

    iter = epoch * len(train_loader)
    for batch_train, _, tags_train, _ in train_loader:
        iter += 1

        # calculate loss
        #batch_train = introduce_random_OOV(batch_train, id_OOV)
        print(tags_train.shape[0])
        pred_train = model(batch_train, tags_train, tags_train.shape[0])
        
        pred_train = pred_train.permute((1, 2, 0))
        tags_train = tags_train.T
        #print(pred_train.shape)
        #print(tags_train.shape)
        loss_train = loss_fun(pred_train, tags_train)
        #print(loss_train)

        # optimization step
        optimizer.zero_grad()
        loss_train.backward()
        """print(model)
        print(model.decoder.linear.weight)
        print(model.decoder.linear)
        print(model.decoder.linear.weight.grad)
        exit()"""
        optimizer.step()

        # calculate validation loss
        batch_val, _, tags_val, _ = next(val_loader.__iter__())
        with torch.no_grad():
            pred_val = model(batch_val, tags_val, tags_val.shape[0])
            pred_val = pred_val.permute((0, 2, 1))
            loss_val = loss_fun(pred_val, tags_val)

        # log progress
        accuracy_train = accuracy(pred_train, tags_train)
        accuracy_val = accuracy(pred_val, tags_val)
        writer.add_scalar('Loss/train', loss_train, iter)
        writer.add_scalar('Loss/validation', loss_val, iter)
        writer.add_scalar('Accuracy/train', accuracy_train, iter)
        writer.add_scalar('Accuracy/validation', accuracy_val, iter)
        print('[{:2d}/{}] Iteration {:4d}: train loss {:6.3f}, val loss {:6.3f}, train accuracy {:6.3}, val accuracy {:6.3}, learning rate {:6.5f}'.format(
            epoch + 1, 
            cfg["epochs"], 
            iter, 
            loss_train, 
            loss_val, 
            accuracy_train,
            accuracy_val,
            cfg['learning_rate']))

cfg = {'learning_rate': 10, 'epochs': 60}
model = Translation(40, 50, vocEng, vocFra)
optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'])
loss_fun = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(cfg['epochs']):
    train(epoch, model, optimizer, train_loader, test_loader, loss_fun, SummaryWriter())
