import torch
from torch.utils.tensorboard import SummaryWriter

import datamaestro
from tqdm import tqdm


def main():
    # Les données supervisées
    x = torch.randn(50, 13)
    y = torch.randn(50, 3)

    # Les paramètres du modèle à optimiser
    w = torch.randn(13, 3,  requires_grad=True)
    b = torch.randn(3,  requires_grad=True)

    # Sets learning rate
    lr = 1e-4
    epsilon = 0.05

    writer = SummaryWriter()
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


if __name__ == '__main__':
    main()
