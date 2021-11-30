import itertools
import logging
from tqdm import tqdm

from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
logging.basicConfig(level=logging.INFO)

ds = prepare_dataset('org.universaldependencies.french.gsd')


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
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

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)


#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)

cfg = {'epochs': 60, 'dim_encode': 30, 'learning_rate': 0.1, 'lr_decay': 0.95, 'prob_oov': 0.05}

class Encoder(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.nb_classes = dim_in

    def forward(self, batch):
        batch_oneHot = nn.functional.one_hot(batch, self.nb_classes)
        return self.linear(batch_oneHot.float())


class Seq2Seq(nn.Module):

    def __init__(self, encoder, lstm):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.lstm = lstm

    def forward(self, batch):
        batch_enc = self.encoder(batch)
        pred, _ = self.lstm(batch_enc)
        return pred

def introduce_random_OOV(batch, id_OOV):
    proba = torch.rand(batch.shape)
    batch[proba < cfg['prob_oov']] = id_OOV
    return batch

def accuracy(batch, tags):
    pred = torch.argmax(batch ,dim=1)
    return torch.mean((pred == tags).float())


def train(epoch, model, optimizer, train_loader, val_loader, loss_fun, writer, id_OOV):

    iter = epoch * len(train_loader)
    for batch_train, tags_train in train_loader:
        iter += 1

        # calculate loss
        batch_train = introduce_random_OOV(batch_train, id_OOV)
        pred_train = model(batch_train)
        pred_train = pred_train.permute((0, 2, 1))
        loss_train = loss_fun(pred_train, tags_train)

        # optimization step
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # calculate validation loss
        batch_val, tags_val = next(val_loader.__iter__())
        with torch.no_grad():
            pred_val = model(batch_val)
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


if __name__ == '__main__':

    writer = SummaryWriter()
    id_OOV = words['__OOV__']
    encoder = Encoder(len(words), cfg['dim_encode'])
    model = Seq2Seq(encoder, nn.LSTM(cfg['dim_encode'], len(tags)))
    loss_fun = nn.CrossEntropyLoss(ignore_index=id_OOV)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'])

    for epoch in range(cfg['epochs']):
        train(epoch, model, optimizer, train_loader, dev_loader, loss_fun, writer, id_OOV)
        #cfg['learning_rate'] *= cfg['lr_decay']
        torch.save(model.state_dict(), 'checkpoints_low/epoch_{}.pt'.format(epoch))



