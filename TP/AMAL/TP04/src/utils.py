import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, input_size, latent_size, output_size):
        super(RNN, self).__init__()
        # Size
        self.inputSize = input_size
        self.latentSize = latent_size
        self.outputSize = output_size
        # Encoder
        self.W_i = nn.Linear(self.inputSize, self.latentSize)
        self.W_h = nn.Linear(self.latentSize, self.latentSize)
        self.encoder_activation = nn.Tanh()
        # Decoder
        self.linearDecode = nn.Linear(self.latentSize, self.outputSize)
        self.funDecode = nn.Softmax(dim=-1)

    def one_step(self, x, h):
        if h is None:
            return self.encoder_activation(self.W_i(x))
        else:
            return self.encoder_activation(self.W_i(x) + self.W_h(h))

    def forward(self, x, h=None):
        h_ = torch.zeros((x.shape[0], x.shape[1], self.latentSize))
        for i in range(x.shape[0]):
            h = self.one_step(x[i, :, :], h)
            h_[i] = h
        return h_

    def decode(self, h):
        return self.funDecode(self.linearDecode(h))


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
            ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
            self.stations_max = torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station


class ForecastMetroDataset(Dataset):
    def __init__(self, data, length=20, stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length = data,length
        if stations_max is None:
            ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
            self.stations_max = torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
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

