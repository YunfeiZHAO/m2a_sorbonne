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
# Téléchargement des données

from datamaestro import prepare_dataset
#
ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()
#
# # Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
# writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#
# # Pour visualiser
# # Les images doivent etre en format Channel (3) x Hauteur x Largeur
# images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1, 3, 1, 1).double()/255.
# # Permet de fabriquer une grille d'images
# images = make_grid(images)
# # Affichage avec tensorboard
# writer.add_image(f'samples', images, 0)
#
#

#  TODO:


class MonDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, item):
        """
        return: a tuple (example, label) correspond to index
        """
        image = self.images[item]
        label = self.labels[item]
        image = image/255.
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        """
        return: the size of data base
        """
        return self.images.shape[0]


class SimpleCustomBatch:
    """ Collate batch and put them in pinning memory to accelerate communication with GPU"""
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


# Saving a model in this way will save the entire module using Python’s pickle module
class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0

# paramters
batch_size = 2
shuffle = False
savepath = Path("model.pch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loader
data = DataLoader(MonDataset(train_images, train_labels), shuffle=shuffle, batch_size=batch_size,
                  collate_fn=collate_wrapper,
                  pin_memory=True)

# training and checkpointing
if savepath.is_file():
    with savepath.open('rb') as fp:
        # restart from saved mode
        state = torch.load(fp)
else:
    pass



