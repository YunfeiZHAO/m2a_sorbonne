from utils import RNN, device, SampleMetroDataset

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context

# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    mse = MSE.apply
    linear = Linear.apply
    loss = mse(linear(x, w, b), y)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    loss.backward()

    ##  TODO:  Mise à jour des paramètres du modèle
    with torch.no_grad():
        w -= epsilon*w.grad
        b -= epsilon*b.grad
        w.grad.zero_()
        b.grad.zero_()