import datetime
import string
import unicodedata
import torch
import sys
from tqdm import tqdm
import logging

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from utils import RNN, AverageMeter, load_yaml

# Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
# Dictionnaire index -> lettre
id2lettre = dict(zip(range(1, len(LETTRES) + 1), LETTRES))
id2lettre[0] = ''  # NULL CHARACTER

# Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))


def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)


def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])


def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


def read_txt(path):
    with open(path, 'r') as file:
        data = file.read().replace('\n', '')
    return data


class TrumpDataset(Dataset):
    def __init__(self, text, maxsent=None, maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip() + "." for p in full_text.split(".") if len(p) > 0]
        if maxsent is not None:
            self.phrases = self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN - t.size(0), dtype=torch.long), t])  # padding with 0 to batch it
        return t[:-1], t[1:]  # last one is the '.'


def train(epoch, data_train, model, loss_fun, optimizer, writer, device):
    loss_meter = AverageMeter()
    iter = epoch * len(data_train)
    model.train()
    for batch, target in data_train:
        batch = torch.transpose(batch, 0, 1).to(device)  # swap the first and second dimension, make it T, B, d=1
        target = torch.transpose(target, 0, 1).to(device)  # T, B
        hidden_outputs = model(batch)  # T, B, h
        # the prediction come from the all hidden layer output
        prediction = model.decode(hidden_outputs)  # T, B, C=96 (probabilities)
        loss = loss_fun(prediction.permute(1, 2, 0), target.permute(1, 0))  # cross_entropy pred: N, C, d1, ... target: N, d1, ...
        loss_meter.update(loss)
        # print("Epoch: {}; Iteration: {}; Loss: {}, Accuracy: {}".format(epoch, iter, loss, acc))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter += 1
        writer.add_scalar("Train/Loss", loss, iter)
    writer.add_scalar("Train/Epoch Loss", loss_meter.avg, epoch)


def generate_phrase(checkpoint_path, first_caracter, sentence_len):
    config = load_yaml('../configs/exo4/trump.yaml')
    # Dimension de l'entrée (1 (in) ou 2 (in/out))
    DIM_INPUT = config.DIM_INPUT
    # Dim latent
    DIM_LATENT = config.DIM_LATENT
    n_letter = len(lettre2id)
    device = 'cpu'
    model = RNN(input_size=DIM_INPUT, latent_size=DIM_LATENT, output_size=n_letter, device=device,
                embedding_size=n_letter, encoder_activation=nn.Tanh(), decoder_activation=None)
    model.load_state_dict(torch.load(checkpoint_path))
    c = string2code(first_caracter).unsqueeze(0)
    predicted_string = ''
    for i in range(sentence_len):
        hidden_outputs = model(c)  # 1, 1
        # the prediction come from the all hidden layer output
        prediction = model.decode(hidden_outputs)  # T, B, C=96 (probabilities)
        c = torch.argmax(prediction, dim=-1)
        s = code2string(c[0])
        predicted_string += s
    print(predicted_string)


def main():
    # config = load_yaml('../configs/exo3/rnn_hangzhou_20.yaml')
    config = load_yaml('../configs/exo4/trump.yaml')
    print(config)
    # max length of a sentence
    maxlen = config.maxlen  # max in data is 809
    # max number of phrase
    maxsent = config.maxsent  # 17066 totally
    # Longueur des séquences
    LENGTH = config.LENGTH
    # Dimension de l'entrée (1 (in) ou 2 (in/out))
    DIM_INPUT = config.DIM_INPUT
    # Dim latent
    DIM_LATENT = config.DIM_LATENT
    # Taille du batch
    BATCH_SIZE = config.BATCH_SIZE

    device = torch.device(config.device)

    PATH = "../data/"

    # Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
    writer = SummaryWriter(f"../experiments/{config.name}-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Dataloader
    text = read_txt('../data/trump_full_speech.txt')
    ds_trump = TrumpDataset(text, maxsent=maxsent, maxlen=maxlen)
    data_train = DataLoader(ds_trump, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    # number of letters in dictionary
    n_letter = len(lettre2id)
    # output with sigmoid is the distribution of letters
    model = RNN(input_size=DIM_INPUT, latent_size=DIM_LATENT, output_size=n_letter, device=device,
                embedding_size=n_letter, encoder_activation=nn.Tanh(), decoder_activation=None)
    model = model.to(device)

    # Optimiser and Loss
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    loss_fun = torch.nn.CrossEntropyLoss()

    try:
        for epoch in tqdm(range(config.epoch)):
            train(epoch, data_train, model, loss_fun, optimizer, writer, device)
        torch.save(model.state_dict(), f'{config.name}.pth')
    except KeyboardInterrupt:
        torch.save(model.state_dict(), f'{config.name}.pth')
        logging.info('Saved interrupt')
        sys.exit(0)



if __name__ == '__main__':
    # main()
    generate_phrase('exo4_rnn_trump.yaml.pth', 't', 50)
    pass