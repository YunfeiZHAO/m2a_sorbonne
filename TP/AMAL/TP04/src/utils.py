import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import yaml

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, input_size, latent_size, output_size, device, embedding_size=None,
                 encoder_activation=nn.Tanh(), decoder_activation=nn.Softmax(dim=-1)):
        super(RNN, self).__init__()
        # Size
        self.inputSize = input_size
        self.latentSize = latent_size
        self.outputSize = output_size
        self.embeddingSize = embedding_size
        # Encoder
        self.W_i = nn.Linear(self.inputSize, self.latentSize)
        self.W_h = nn.Linear(self.latentSize, self.latentSize)
        self.encoder_activation = encoder_activation
        # Decoder
        self.linearDecode = nn.Linear(self.latentSize, self.outputSize)
        self.decoder_activation = decoder_activation
        self.device = device
        # Embedding
        if self.embeddingSize:
            self.embedding = nn.Linear(self.embeddingSize, self.inputSize)

    def one_step(self, x, h):
        """
        :param x: B, input_size
        :param h: B,latent_size
        """
        if h is None:
            return self.encoder_activation(self.W_i(x))
        else:
            return self.encoder_activation(self.W_i(x) + self.W_h(h))

    def forward(self, x, h=None):
        """
        :param x: sequence of input: T, B, input_size
        :param h: initial hidden input: B,latent_size
        """
        if self.embeddingSize:
            x = F.one_hot(x, self.embeddingSize).type(torch.float)
            x = self.embedding(x)
        hidden_size = list(x.size())
        hidden_size[-1] = self.latentSize
        self.register_buffer('hidden_outputs', torch.zeros(tuple(hidden_size), device=self.device), persistent=False)

        for i, x_t in enumerate(x):
            h = self.one_step(x_t, h)
            self.hidden_outputs[i] = h
        return self.hidden_outputs

    def decode(self, h):
        """ we may not need activation function, because nn.CrossEntropyLoss has softmax
        :param h: a given batch of hidden output: B, latent_size
        """
        if self.decoder_activation is None:
            return self.linearDecode(h)
        else:
            return self.decoder_activation(self.linearDecode(h))


class SampleMetroDataset(Dataset):
    def __init__(self, data, length=20, stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length = data, length
        self.stations_max = stations_max
        if self.stations_max is None:
            # Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant of different stations
            self.stations_max = torch.max(self.data.view(-1, self.data.size(2), self.data.size(3)), 0)[0]
        # Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        # longueur en fonction de la longueur considérée des séquences
        # si le sequence est plus long, le nombre de sample sera plus petit
        return self.classes * self.nb_days * (self.nb_timeslots - self.length)

    def __getitem__(self, i):
        """ transformation de l'index 1d vers une indexation 3d (station, timeslot, day)
        :return: renvoie une séquence de longueur length et l'id de la station.
        """
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)  # take value from 0 to classes
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)

        timeslot = i // self.nb_days  # take value from 0 to nb_timeslots - length (53 if length = 20)
        day = i % self.nb_days  # take value from 0-18 for train
        return self.data[day, timeslot:(timeslot+self.length), station], station


class ForecastMetroDataset(Dataset):
    def __init__(self, data, length=20, stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length = data, length
        self.stations_max = stations_max
        if self.stations_max is None:
            ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
            self.stations_max = torch.max(self.data.view(-1, self.data.size(2), self.data.size(3)), 0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self, i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day, timeslot:(timeslot+self.length-1)], self.data[day, (timeslot+1):(timeslot+self.length)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, keep_all=False):
        self.reset()
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.data is not None:
            self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DotDict(dict):
    """dot.notation access to dictionary attributes (Thomas Robert)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_yaml(path):
    with open(path, 'r') as stream:
        opt = yaml.load(stream, Loader=yaml.Loader)
    return DotDict(opt)