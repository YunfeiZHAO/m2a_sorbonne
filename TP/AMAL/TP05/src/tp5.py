import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import yaml
import datetime
from tqdm import tqdm
import logging

from textloader import *
from generate import *


#  TODO:
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


def read_txt(path):
    with open(path, 'r') as file:
        data = file.read().replace('\n', '')
    return data


def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """ Implémentation maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    output = output.permute((1, 2, 0))  # B, d, L
    target = target.permute((1, 0))  # B, L
    loss = F.cross_entropy(output, target, reduction='none')  # get element wise cross entropy loss: L, B
    mask = target != padcar
    loss = torch.sum(loss * mask)/torch.sum(mask)
    return loss


class RNN(nn.Module):
    def __init__(self, input_size, latent_size, output_size, device, num_embeddings=None, encoder_activation=nn.Tanh()):
        super(RNN, self).__init__()
        # Size
        self.inputSize = input_size
        self.latentSize = latent_size
        self.outputSize = output_size
        self.num_embeddings = num_embeddings
        # Encoder
        self.W_i = nn.Linear(self.inputSize, self.latentSize)
        self.W_h = nn.Linear(self.latentSize, self.latentSize)
        self.encoder_activation = encoder_activation
        # Decoder
        self.linearDecode = nn.Linear(self.latentSize, self.outputSize)
        self.device = device
        # Embedding
        if self.num_embeddings:
            self.embedding = nn.Embedding(num_embeddings, input_size)

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
        if self.num_embeddings:
            x = self.embedding(x)
        hidden_size = list(x.size())
        hidden_size[-1] = self.latentSize
        self.register_buffer('hidden_outputs', torch.zeros(tuple(hidden_size), device=self.device), persistent=False)

        for i, x_t in enumerate(x):
            h = self.one_step(x_t, h)
            self.hidden_outputs[i] = h
        return self.hidden_outputs

    def decode(self, h, decoder_activation=None):
        """ we may not need activation function, because nn.CrossEntropyLoss has softmax
        :param h: a given batch of hidden output: B, latent_size
        :param decoder_activation: activation function for decoder
        """
        if decoder_activation is None:
            return self.linearDecode(h)
        else:
            return decoder_activation(self.linearDecode(h))


class LSTM(nn.Module):
    def __init__(self, input_size, latent_size, output_size, device, num_embeddings=None):
        super(LSTM, self).__init__()
        # size
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.num_embeddings = num_embeddings

        # Encoder
        # # weight, here we let memory has the same dimension as latent state
        self.W_f = nn.Linear(self.latent_size + self.input_size, self.latent_size)  # forget gate weight
        self.W_i = nn.Linear(self.latent_size + self.input_size, self.latent_size)  # input gate weight
        self.W_c = nn.Linear(self.latent_size + self.input_size, self.latent_size)  # cell state gate weight
        self.W_o = nn.Linear(self.latent_size + self.input_size, self.latent_size)  # output gate weight
        # # activation
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Decoder
        self.linearDecode = nn.Linear(self.latent_size, self.output_size)
        self.device = device

        # Embedding
        if self.num_embeddings:
            self.embedding = nn.Embedding(num_embeddings, input_size)

    def one_step(self, x, h, c):
        """
        :param x: B, input_size
        :param h: B,latent_size
        """
        if c is None:
            c = torch.zeros((x.size()[0], self.latent_size), device=self.device)
        if h is None:
            h = torch.zeros((x.size()[0], self.latent_size), device=self.device)
            hx = torch.cat((h, x), dim=-1)
        else:
            hx = torch.cat((h, x), dim=-1)

        # forget gate
        f = self.sigmoid(self.W_f(hx))
        # input gate
        i = self.sigmoid(self.W_i(hx))
        # cell state gate
        c_hat = self.tanh(self.W_c(hx))
        # cell state update
        new_c = f * c + i * c_hat
        # output gate
        o = self.sigmoid(self.W_o(hx))
        new_h = o * self.tanh(new_c)
        return new_h, new_c

    def forward(self, x, h=None, c=None):
        """
        :param x: sequence of input: T, B, input_size
        :param h: initial hidden input: B,latent_size
        """
        if self.num_embeddings:
            x = self.embedding(x)
        hidden_size = list(x.size())
        hidden_size[-1] = self.latent_size
        self.register_buffer('hidden_outputs', torch.zeros(tuple(hidden_size), device=self.device), persistent=False)

        for i, x_t in enumerate(x):
            h, c = self.one_step(x_t, h, c)
            self.hidden_outputs[i] = h
        return self.hidden_outputs

    def decode(self, h, decoder_activation=None):
        """ we may not need activation function, because nn.CrossEntropyLoss has softmax
        :param h: a given batch of hidden output: B, latent_size
        :param decoder_activation: activation function for decoder
        """
        if decoder_activation is None:
            return self.linearDecode(h)
        else:
            return decoder_activation(self.linearDecode(h))


class GRU(nn.Module):
    def __init__(self, input_size, latent_size, output_size, device, num_embeddings=None):
        super(GRU, self).__init__()
        # size
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.num_embeddings = num_embeddings

        # Encoder
        # # weight
        self.W_z = nn.Linear(self.latent_size + self.input_size, self.latent_size)  # update gate weight
        self.W_r = nn.Linear(self.latent_size + self.input_size, self.latent_size)  # reset gate weight
        self.W_h = nn.Linear(self.latent_size + self.input_size, self.latent_size)  # latent weight
        # # activation
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Decoder
        self.linearDecode = nn.Linear(self.latent_size, self.output_size)
        self.device = device

        # Embedding
        if self.num_embeddings:
            self.embedding = nn.Embedding(num_embeddings, input_size)

    def one_step(self, x, h):
        """
        :param x: B, input_size
        :param h: B,latent_size
        """
        if h is None:
            h = torch.zeros((x.size()[0], self.latent_size), device=self.device)
            hx = torch.cat((h, x), dim=-1)
        else:
            hx = torch.cat((h, x), dim=-1)
        z = self.sigmoid(self.W_z(hx))
        r = self.sigmoid(self.W_r(hx))
        new_h = (1 - z) * h + z * self.tanh(self.W_h(torch.cat((r * h, x), dim=-1)))
        return new_h

    def forward(self, x, h=None):
        """
        :param x: sequence of input: T, B, input_size
        :param h: initial hidden input: B,latent_size
        """
        if self.num_embeddings:
            x = self.embedding(x)
        hidden_size = list(x.size())
        hidden_size[-1] = self.latent_size
        self.register_buffer('hidden_outputs', torch.zeros(tuple(hidden_size), device=self.device), persistent=False)

        for i, x_t in enumerate(x):
            h = self.one_step(x_t, h)
            self.hidden_outputs[i] = h
        return self.hidden_outputs

    def decode(self, h, decoder_activation=None):
        """ we may not need activation function, because nn.CrossEntropyLoss has softmax
        :param h: a given batch of hidden output: B, latent_size
        :param decoder_activation: activation function for decoder
        """
        if decoder_activation is None:
            return self.linearDecode(h)
        else:
            return decoder_activation(self.linearDecode(h))


#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
def train(epoch, data_train, model, loss_fun, optimizer, writer, device):
    loss_meter = AverageMeter()
    iter = epoch * len(data_train)
    model.train()
    for batch in data_train:
        input = batch[:-1, ].to(device)  # T, B
        target = batch[1:, ].to(device)  # T, B
        hidden_outputs = model(input)  # T, B, h
        # the prediction come from the all hidden layer output
        prediction = model.decode(hidden_outputs)  # T, B, C=96 (probabilities)

        # Loss Function
        loss = loss_fun(prediction, target, PAD_IX)  # PAD_IX from textloader.py
        loss_meter.update(loss)

        # Optimiser
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter += 1
        writer.add_scalar("Train/Loss", loss, iter)
    # show weights and gradients at the end of each epoch
    for tag, value in model.named_parameters():
        tag = tag.replace('/', '.')
        writer.add_histogram('Weights/' + tag, value.data.cpu(), epoch)
        writer.add_histogram('Gradients/' + tag, value.grad.data.cpu(), epoch)

    writer.add_scalar("Train/Epoch Loss", loss_meter.avg, epoch)


def main():
    # config = load_yaml('../configs/exo4/trump.yaml')
    # config = load_yaml('../configs/exo4/GNU_trump.yaml')
    config = load_yaml('../configs/exo4/LSTM_trump.yaml')
    print(config)
    # max length of a sentence
    maxlen = config.maxlen  # max in data is 809
    # max number of phrase
    maxsent = config.maxsent  # 17066 totally
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
    ds_trump = TextDataset(text, maxsent=maxsent, maxlen=maxlen)
    data_train = DataLoader(ds_trump, collate_fn=pad_collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # Model
    # number of letters in dictionary
    n_letter = len(lettre2id)
    # output with sigmoid is the distribution of letters
    if config.model == 'rnn':
        model = RNN(input_size=DIM_INPUT, latent_size=DIM_LATENT, output_size=n_letter, device=device,
                    num_embeddings=n_letter,
                    encoder_activation=nn.Tanh())
    elif config.model == 'gnu':
        model = GRU(input_size=DIM_INPUT, latent_size=DIM_LATENT, output_size=n_letter, device=device,
                    num_embeddings=n_letter)
    elif config.model == 'lstm':
        model = LSTM(input_size=DIM_INPUT, latent_size=DIM_LATENT, output_size=n_letter, device=device,
                     num_embeddings=n_letter)
    print(model)
    model = model.to(device)

    # Optimiser and Loss
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    loss_fun = maskedCrossEntropy

    try:
        for epoch in tqdm(range(config.epoch)):
            train(epoch, data_train, model, loss_fun, optimizer, writer, device)
        torch.save(model.state_dict(), f'{config.name}.pth')
    except KeyboardInterrupt:
        torch.save(model.state_dict(), f'{config.name}.pth')
        logging.info('Saved interrupt')
        sys.exit(0)


def generate_sequence(config_path, checkpoint_path, generator, eos=1, start="", maxlen=200):
    # config = load_yaml('../configs/exo4/trump.yaml')
    # config = load_yaml('../configs/exo4/GNU_trump.yaml')
    # config = load_yaml('../configs/exo4/LSTM_trump.yaml')
    config = load_yaml(config_path)
    print(config)

    # Dimension de l'entrée (1 (in) ou 2 (in/out))
    DIM_INPUT = config.DIM_INPUT
    # Dim latent
    DIM_LATENT = config.DIM_LATENT

    device = torch.device('cpu')

    # Model
    # number of letters in dictionary
    n_letter = len(lettre2id)

    if config.model == 'rnn':
        model = RNN(input_size=DIM_INPUT, latent_size=DIM_LATENT, output_size=n_letter, device=device,
                    num_embeddings=n_letter,
                    encoder_activation=nn.Tanh())
    elif config.model == 'gnu':
        model = GRU(input_size=DIM_INPUT, latent_size=DIM_LATENT, output_size=n_letter, device=device,
                    num_embeddings=n_letter)
    elif config.model == 'lstm':
        model = LSTM(input_size=DIM_INPUT, latent_size=DIM_LATENT, output_size=n_letter, device=device,
                     num_embeddings=n_letter)
    print(model)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()
    if generator == 'multinomial':
        phrases = generate(rnn=model, eos=eos, start=start, maxlen=maxlen)
    if generator == 'beam':
        phrases = generate_beam(rnn=model, eos=eos, k=10, start=start, maxlen=maxlen)

    return phrases


if __name__ == '__main__':
    # main()
    # generate_sequence(config_path='../configs/exo4/trump.yaml', checkpoint_path='./exo4_rnn_trump.yaml.pth',
    #                   generator='multinomial', eos=1, start="", maxlen=200)
    # generate_sequence(config_path='../configs/exo4/GNU_trump.yaml', checkpoint_path='./exo5_gnu_trump.yaml.pth',
    #                   generator='multinomial', eos=1, start="f", maxlen=200)

    # generate_sequence(config_path='../configs/exo4/LSTM_trump.yaml', checkpoint_path='./exo5_lstm_trump.yaml.pth',
    #                   generator='multinomial', eos=1, start="", maxlen=200)

    # generate_sequence(config_path='../configs/exo4/trump.yaml', checkpoint_path='./exo4_rnn_trump.yaml.pth',
    #                   generator='beam', eos=1, start="f", maxlen=200)

    # generate_sequence(config_path='../configs/exo4/trump.yaml', checkpoint_path='./exo4_rnn_trump.yaml.pth',
    #                   generator='beam', eos=1, start="f", maxlen=200)
    # generate_sequence(config_path='../configs/exo4/GNU_trump.yaml', checkpoint_path='./exo5_gnu_trump.yaml.pth',
    #                   generator='beam', eos=1, start="o", maxlen=50)
    # generate_sequence(config_path='../configs/exo4/LSTM_trump.yaml', checkpoint_path='./exo5_lstm_trump.yaml.pth',
    #                   generator='beam', eos=1, start="k", maxlen=50)

    generate_sequence(config_path='../configs/exo4/LSTM_trump.yaml', checkpoint_path='./exo5_lstm_trump.yaml.pth',
                      generator='multinomial', eos=1, start="", maxlen=200)

    # generate_sequence(config_path='../configs/exo4/trump.yaml', checkpoint_path='./exo4_rnn_trump.yaml.pth',
    #                   generator='beam', eos=1, start="", maxlen=200)

    pass

