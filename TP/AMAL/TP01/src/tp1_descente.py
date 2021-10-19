import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


def gradient_descent():
    # Les données supervisées
    x = torch.randn(50, 13)
    y = torch.randn(50, 3)

    # Les paramètres du modèle à optimiser
    w = torch.randn(13, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    lr = 0.01

    writer = SummaryWriter('../../hm1_results/TP01 my class')
    for n_iter in range(100):
        ##  TODO:  Calcul du forward (loss)
        mse = MSE.apply
        lin = Linear.apply
        # `loss` doit correspondre au coût MSE calculé à cette itération
        loss = mse(lin(x, w, b), y)
        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train', loss, n_iter)
        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss}")

        ##  TODO:  Calcul du backward (grad_w, grad_b)
        loss.backward()

        ##  TODO:  Mise à jour des paramètres du modèle
        with torch.no_grad():
            w -= w.grad*lr
            b -= b.grad*lr
            w.grad.zero_()
            b.grad.zero_()

if __name__ == '__main__':
    gradient_descent()

