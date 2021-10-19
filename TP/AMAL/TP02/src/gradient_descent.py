import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from typing import OrderedDict

import datamaestro
from tqdm import tqdm

import pandas as pd
import numpy as np


def regression_gradient_descent():
    # Les données supervisées
    x = torch.randn(50, 13)
    y = torch.randn(50, 3)

    # Les paramètres du modèle à optimiser
    w = torch.randn(13, 3,  requires_grad=True)
    b = torch.randn(3,  requires_grad=True)

    # Sets learning rate
    lr = 1e-4
    epsilon = 0.05

    writer = SummaryWriter('simple regression')
    for n_iter in range(500):

        #  TODO:  Calcul du forward (loss)
        y_hat = x.mm(w) + b
        loss = torch.sum((y - y_hat)**2)

        # faut verifier toujours le shape
        print(y_hat.size())
        print((y-y_hat).size())


        # `loss` doit correspondre au coût MSE calculé à cette itération
        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train', loss, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss}")

        #  TODO:  Calcul du backward (grad_w, grad_b)
        loss.backward()

        #  TODO:  Mise à jour des paramètres du modèle
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        ### ici on peut aussi ecrire comme cela
        # w.data -= lr * w.grad
        # b.data -= lr * b.grad
        ### si on accès une variable par .data, elle ne sera pas ajoutée dans le graphe

        # sinon le gradient s'accumule
        w.grad.zero_()
        b.grad.zero_()


def boston_housing_gradient_descent():
    def model(x):
        return x @ w + b

    def mse(t1, t2):
        diff = t1 - t2
        return torch.sum(diff * diff)/diff.numel()

    data = pd.read_csv('./housing.data', header=None, delimiter=r"\s+")
    # Les données supervisées
    n = 400
    X_train = torch.tensor(data.iloc[0:n, 0:13].values, dtype=torch.float)
    y_train = torch.tensor(data.iloc[0:n, 13].values, dtype=torch.float).unsqueeze(dim=-1)
    X_test = torch.tensor(data.iloc[n:, 0:13].values, dtype=torch.float)
    y_test = torch.tensor(data.iloc[n:, 13].values, dtype=torch.float).unsqueeze(dim=-1)

    # Les paramètres du modèle à optimiser
    w = torch.randn(13, 1,  requires_grad=True)
    b = torch.randn(1,  requires_grad=True)

    # Sets learning rate
    lr = 1e-6
    epsilon = 0.05

    writer = SummaryWriter('Boston house regression')
    for n_iter in range(500):
        #  TODO:  Calcul du forward (loss)
        y_hat = model(X_train)
        loss_train = mse(y_train, y_hat)

        writer.add_scalar('Train/MSE', loss_train, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: loss_train {loss_train}")

        #  TODO:  Calcul du backward (grad_w, grad_b)
        loss_train.backward()

        # For test
        y_test_hat = model(X_test)
        loss_test = mse(y_test_hat, y_test)
        writer.add_scalar('Test/MSE', loss_test, n_iter)

        #  TODO:  Mise à jour des paramètres du modèle
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()



def boston_housing_gradient_descent_minibatch():
    def model(x):
        return x @ w + b

    def mse(t1, t2):
        diff = t1 - t2
        return torch.sum(diff * diff)/diff.numel()

    def se(t1, t2):
        diff = t1 - t2
        return torch.sum(diff * diff)

    data = pd.read_csv('./housing.data', header=None, delimiter=r"\s+")
    # Les données supervisées
    n = 400
    X_train = torch.tensor(data.iloc[0:n, 0:13].values, dtype=torch.float)
    y_train = torch.tensor(data.iloc[0:n, 13].values, dtype=torch.float).unsqueeze(dim=-1)
    X_test = torch.tensor(data.iloc[n:, 0:13].values, dtype=torch.float)
    y_test = torch.tensor(data.iloc[n:, 13].values, dtype=torch.float).unsqueeze(dim=-1)

    # Les paramètres du modèle à optimiser
    w = torch.randn(13, 1,  requires_grad=True)
    b = torch.randn(1,  requires_grad=True)

    # Sets learning rate
    lr = 1e-6
    epsilon = 0.05
    batch_size = 30

    writer = SummaryWriter('Boston house regression with mini batch')
    for n_iter in range(500):
        start_index = 0
        loss_train = 0
        while start_index < n:
            X_train_batch = X_train[start_index: start_index+batch_size]
            y_train_batch = y_train[start_index: start_index+batch_size]
            start_index += batch_size

            y_hat_batch = model(X_train_batch)
            sse = se(y_train_batch, y_hat_batch)
            loss_train_batch = sse/batch_size

            loss_train_batch.backward()
            #  TODO:  Mise à jour des paramètres du modèle
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
            # sinon le gradient s'accumule
            w.grad.zero_()
            b.grad.zero_()

            loss_train += sse

        writer.add_scalar('Train/MSE', loss_train/n, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: loss_train {loss_train}")


        # For test
        y_test_hat = model(X_test)
        loss_test = mse(y_test_hat, y_test)
        writer.add_scalar('Test/MSE', loss_test, n_iter)


class My_NN(nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super(My_NN, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid)
        self.act1 = nn.Tanh()
        self.linear2 = nn.Linear(hid, out_dim)

    def forward(self, X):
        lin1 = self.linear1(X)
        act1 = self.act1(lin1)
        lin2 = self.linear2(act1)
        return lin2


def My_NN_GD():
    writer = SummaryWriter('Boston house regression with my nn')
    data = pd.read_csv('./housing.data', header=None, delimiter=r"\s+")
    # Les données supervisées
    n = 400
    X_train = torch.tensor(data.iloc[0:n, 0:13].values, dtype=torch.float)
    y_train = torch.tensor(data.iloc[0:n, 13].values, dtype=torch.float).unsqueeze(dim=-1)
    X_test = torch.tensor(data.iloc[n:, 0:13].values, dtype=torch.float)
    y_test = torch.tensor(data.iloc[n:, 13].values, dtype=torch.float).unsqueeze(dim=-1)

    my_nn = My_NN(13, 50, 1)
    lr = 1e-6
    optim = torch.optim.SGD(my_nn.parameters(), lr=lr)
    optim.zero_grad()
    for n_iter in range(1000):
        y_train_pred = my_nn(X_train)
        mean_error_train = nn.MSELoss()(y_train, y_train_pred)
        mean_error_train.backward()
        optim.step()
        optim.zero_grad()
        writer.add_scalar('Train/MSE', mean_error_train, n_iter)
        print(f'Loss train: {mean_error_train}')
        y_test_hat = my_nn(X_test)
        loss_test = nn.MSELoss()(y_test_hat, y_test)
        writer.add_scalar('Test/MSE', loss_test, n_iter)


def container():
    writer = SummaryWriter('Boston house regression with container')
    data = pd.read_csv('./housing.data', header=None, delimiter=r"\s+")
    # Les données supervisées
    n = 400
    in_dim = 13
    hid = 50
    out_dim = 1
    X_train = torch.tensor(data.iloc[0:n, 0:13].values, dtype=torch.float)
    y_train = torch.tensor(data.iloc[0:n, 13].values, dtype=torch.float).unsqueeze(dim=-1)
    X_test = torch.tensor(data.iloc[n:, 0:13].values, dtype=torch.float)
    y_test = torch.tensor(data.iloc[n:, 13].values, dtype=torch.float).unsqueeze(dim=-1)

    net = nn.Sequential(OrderedDict([
        ('Lin1', nn.Linear(in_dim, hid)),
        ('Tanh', nn.Tanh()),
        ('Lin2', nn.Linear(hid, out_dim))
    ]))

    lr = 1e-6
    optim = torch.optim.SGD(net.parameters(), lr=lr)
    optim.zero_grad()
    for n_iter in range(1000):
        y_train_pred = net(X_train)
        mean_error_train = nn.MSELoss()(y_train, y_train_pred)
        mean_error_train.backward()
        optim.step()
        optim.zero_grad()
        writer.add_scalar('Train/MSE', mean_error_train, n_iter)
        print(f'Loss train: {mean_error_train}')
        y_test_hat = net(X_test)
        loss_test = nn.MSELoss()(y_test_hat, y_test)
        writer.add_scalar('Test/MSE', loss_test, n_iter)


def main():
    # regression_gradient_descent()
    # boston_housing_gradient_descent()
    # boston_housing_gradient_descent_minibatch()
    # My_NN_GD()
    container()

if __name__ == '__main__':
    main()
