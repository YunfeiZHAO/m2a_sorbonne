from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime


from datamaestro import prepare_dataset

# # Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
# writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#
# # Pour visualiser
# # Les images doivent etre en format Channel (3) x Hauteur x Largeur
# images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# # Permet de fabriquer une grille d'images
# images = make_grid(images)
# # Affichage avec tensorboard
# writer.add_image(f'samples', images, 0)


class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        im = self.images[index].flatten()/255.
        label = self.labels[index]
        return torch.tensor(im, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return self.images.shape[0]


class MyAutoencodeur(nn.Module):
    def __init__(self, input_dim, hidden_dim1):
        super(MyAutoencodeur, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim1, input_dim)
        self.act2 = nn.Sigmoid()
        self.linear2.weight = nn.Parameter(self.linear1.weight.T)

    def forward(self, X):
        # encoder
        lin1 = self.linear1(X)
        projection = self.act1(lin1)
        # decoder
        lin2 = self.linear2(projection)
        Xhat = self.act2(lin2)

        return projection, Xhat

class MyAutoencodeurMoreLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hiden_dim2):
        super(MyAutoencodeurMoreLayers, self).__init__()
        # Encoder
        self.en_linear1 = nn.Linear(input_dim, hidden_dim1)
        self.en_act1 = nn.ReLU()
        self.en_linear2 = nn.Linear(hidden_dim1, hiden_dim2)
        self.en_act2 = nn.ReLU()
        # Decoder
        self.de_linear1 = nn.Linear(hiden_dim2, hidden_dim1)
        self.de_act1 = nn.ReLU()
        self.de_linear2 = nn.Linear(hidden_dim1, input_dim)
        self.act2 = nn.Sigmoid()

        self.de_linear1.weight = nn.Parameter(self.en_linear2.weight.T)
        self.de_linear2.weight = nn.Parameter(self.en_linear1.weight.T)

    def forward(self, X):
        # encoder
        en_lin1 = self.en_linear1(X)
        en_a_lin1 = self.en_act1(en_lin1)
        en_lin2 = self.en_linear2(en_a_lin1)
        projection = self.en_act2(en_lin2)
        # decoder
        de_lin1 = self.de_linear1(projection)
        de_a_lin1 = self.de_act1(de_lin1)
        de_lin2 = self.de_linear2(de_a_lin1)
        Xhat = self.act2(de_lin2)
        return projection, Xhat


def load_data():
    ds = prepare_dataset("com.lecun.mnist")
    train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
    test_images, test_labels = ds.test.images.data(), ds.test.labels.data()
    return train_images, train_labels, test_images, test_labels

def train_model():
    # Dataloader
    train_images, train_labels, test_images, test_labels = load_data()
    BATCH_SIZE = 300
    data = DataLoader(MyDataset(train_images, train_labels), shuffle=True, batch_size=BATCH_SIZE)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = MyAutoencodeur(input_dim=28*28, hidden_dim1=100)
    model = MyAutoencodeurMoreLayers(input_dim=28*28, hidden_dim1=300, hiden_dim2=100)
    model.to(device)

    # Optimiser
    lr = 1e-6
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    optim.zero_grad()

    # Visualisation
    writer = SummaryWriter(log_dir="../../hm1_results/auto-encoder_4layers")

    # gradient descent
    epoch = 100
    for n_iter in range(epoch):
        for x, y in data:
            x = x.to(device)
            projection, x_hat = model(x)
            mse = nn.MSELoss()
            loss = mse(x, x_hat)
            loss.backward()
            optim.step()
            optim.zero_grad()
        # last loss of a batch in an interation
        writer.add_scalar('X/MSE', loss, n_iter)
        if n_iter % 10 == 0:
            print(f'Iteration: {n_iter} MSE: {loss}')

    im = x.view(BATCH_SIZE, 28, 28).unsqueeze(1)
    writer.add_embedding(mat=projection, label_img=im, global_step=n_iter)
    torch.save({
                'epoch': n_iter,
                'model_state_dict': model.state_dict()},
                Path='./four_layers.pth')


if __name__ == '__main__':
   pass