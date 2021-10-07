import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm


writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax, dtype=torch.float)
datay = torch.tensor(datay, dtype=torch.float).reshape(-1, 1)

# TODO:

a = torch.rand((1, 10), requires_grad=True)
b = torch.rand((1, 10), requires_grad=True)
c = a.mm(b.t())
d = 2 * c
c.retain_grad()  # on veut conserver le gradient par rapport à c
d.backward()  ## calcul du gradient et retropropagation

##jusqu’aux feuilles du graphe de calcul
print(d.grad)  # Rien : le gradient par rapport à d n’est pas conservé
print(c.grad)  # Celui-ci est conservé
print(a.grad)  # gradient de d par rapport à a qui est une feuille
print(b.grad)  # gradient de d par rapport à b qui est une feuille
d = 2 * a.mm(b.t())
d.backward()
print(a.grad)  ## 2 fois celui d’avant, le gradient est additioné
a.grad.data.zero_()  ## reinitialisation du gradient pour a
d = 2 * a.mm(b.t())
d.backward()
print(a.grad)  ## Cette fois, c’est ok
with torch.no_grad():
    c = a.mm(b.t())  ## Le calcul est effectué sans garder le graphe de calcul
    c.backward()  ## Erreur
