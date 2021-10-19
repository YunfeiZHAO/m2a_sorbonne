import torch
from tp1 import mse, linear

#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)
if __name__ == '__main__':
    yhat = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
    y = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
    result = torch.autograd.gradcheck(mse, (yhat, y))
    print(result)
