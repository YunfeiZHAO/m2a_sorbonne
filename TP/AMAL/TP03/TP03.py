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
        return im, label

    def __len__(self):
        return self.images.shape[0]


def load_data():
    ds = prepare_dataset("com.lecun.mnist")
    train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
    test_images, test_labels = ds.test.images.data(), ds.test.labels.data()
    return train_images, train_labels, test_images, test_labels



train_images, train_labels, test_images, test_labels = load_data()
BATCH_SIZE = 30
data = DataLoader(MyDataset(train_images, train_labels), shuffle=True, batch_size=BATCH_SIZE)
for x, y in data:
    break
