from tqdm import tqdm
import datetime

from torch.nn.modules import loss
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from utils import RNN, ForecastMetroDataset, AverageMeter, load_yaml


#  TODO:  Question 3 : Prédiction de séries temporelles


def train(epoch, data_train, model, loss_fun, optimizer, writer):
    loss_meter = AverageMeter()
    iter = epoch*len(data_train)
    model.train()
    for batch, target in data_train:
        batch = torch.transpose(batch, 0, 1).to(device)  # swap the first and second dimension, make it T, B, c, d
        target = torch.transpose(target, 0, 1).to(device)  # T, B, c, d
        hidden_outputs = model(batch)  # T, B, h
        prediction = model.decode(hidden_outputs)  # the prediction come from the all hidden layer output: T, B, c, d
        loss = loss_fun(prediction, target)
        loss_meter.update(loss)
        # print("Epoch: {}; Iteration: {}; Loss: {}, Accuracy: {}".format(epoch, iter, loss, acc))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter += 1
        writer.add_scalar("Train/Loss", loss, iter)
    writer.add_scalar("Train/Epoch Loss", loss_meter.avg, epoch)


def evaluate(epoch, data_test, model, loss_fun, writer):
    model.eval()
    loss_meter = AverageMeter()
    iter = epoch*len(data_test)
    for batch, target in data_test:
        batch = torch.transpose(batch, 0, 1).to(device)  # swap the first and second dimension, make it T, B, c, d
        target = torch.transpose(target, 0, 1).to(device)  # T, B, d
        hidden_outputs = model(batch)  # T, B, h
        prediction = model.decode(hidden_outputs)  # the prediction come from the all hidden layer output: T, B, c, d
        loss = loss_fun(prediction, target)
        loss_meter.update(loss)
        iter += 1
        # print("Epoch: {}; Iteration: {}; Loss: {}, Accuracy: {}".format(epoch, iter, loss, acc))
        writer.add_scalar("Test/Loss", loss, iter)
    writer.add_scalar("Test/Epoch Loss", loss_meter.avg, epoch)


def accuracy(prediction, target):
    prediction = torch.argmax(prediction, dim=1)
    return torch.mean((prediction == target).float())


if __name__ == '__main__':
    #config = load_yaml('../configs/exo3/rnn_hangzhou_20.yaml')
    config = load_yaml('../configs/exo3/rnn_hangzhou_20_1dim.yaml')
    print(config)
    # Nombre de stations utilisé
    CLASSES = config.CLASSES
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

    matrix_train, matrix_test = torch.load(open(PATH + "hzdataset.pch", "rb"))
    ds_train = ForecastMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
    ds_test = ForecastMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
    data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

    model = RNN(input_size=DIM_INPUT, latent_size=DIM_LATENT, output_size=DIM_INPUT, device=device,
                encoder_activation=nn.Tanh(), decoder_activation=None)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    loss_fun = torch.nn.MSELoss()

    for epoch in tqdm(range(config.epoch)):
        train(epoch, data_train, model, loss_fun, optimizer, writer)
        evaluate(epoch, data_test, model, loss_fun, writer)
