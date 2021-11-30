from torch.nn.modules import loss
from utils import RNN, device, SampleMetroDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "../../data/"

matrix_train, matrix_test = torch.load(open(PATH + "hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence


def train(epoch, data_train, model, loss_fun, optimizer, writer):
    iter = epoch*len(data_train)
    for batch, target in data_train:
        batch = torch.transpose(batch, 0, 1)
        prediction = model(batch)
        prediction = model.decode(prediction[-1, :, :])
        loss = loss_fun(prediction, target)
        acc = accuracy(prediction, target)
        print("Epoch: {}; Iteration: {}; Loss: {}, Accuracy: {}".format(epoch, iter, loss, acc))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter += 1
        writer.add_scalar("Loss", loss, iter)
        writer.add_scalar("Accuracy", acc, iter)

def accuracy(prediction, target):
    prediction = torch.argmax(prediction, dim=1)
    return torch.mean((prediction==target).float())

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


model = RNN(DIM_INPUT, 100, CLASSES)
optimizer = torch.optim.SGD(model.parameters(), 0.01)
loss_fun = torch.nn.CrossEntropyLoss()

epochs = 1000

for epoch in range(epochs):
    train(epoch, data_train, model, loss_fun, optimizer, writer)
"""
i = 0
for x, y in data_train:
    x = torch.transpose(x, 0, 1)
    print(x.shape)
    print(x)
    print(y.shape)
    print(y)
    i += 1
    if i == 1:
        break
    exit()
"""

#print(data_train.shape)
#print(ds_train[1])
#nn = RNN(10, )

