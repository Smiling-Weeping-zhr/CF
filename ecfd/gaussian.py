import torch
import numpy as np
from typing import Union

def gaussian_ecfd(X: torch.Tensor, Y: torch.Tensor, sigmas: Union[list, torch.Tensor]= [[1.0], None], 
                  optimize_sigma: bool = False) -> torch.Tensor:
    """
    Gaussian ECDF distance between two distributions.
    :param X: (N, D) tensor
    :param Y: (N, D) tensor
    :param sigmas: list or torch.Tensor -- a list of floats or a torch 
                   Tensor of shape [1 x D] if optimize_sigma is True
    :param optimize_sigma: bool
    :return: ECDF distance
    """
    total_loss = 0.0
    if not optimize_sigma:
        for sigma in sigmas:
            batch_loss = _gaussian_ecfd(X, Y, sigma)
            total_loss += batch_loss
    else:
        batch_loss = _gaussian_ecfd(X, Y, sigmas)
        total_loss = batch_loss / torch.norm(sigmas, p=2)
    return total_loss
            
def _gaussian_ecfd(X: torch.Tensor, Y: torch.Tensor, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Gaussian ECDF distance between two distributions.
    :param X: (N, D) tensor
    :param Y: (N, D) tensor
    :param sigma: float
    :return: ECDF loss
    """
    num_freqs = 4096
    wX, wY = 1.0, 1.0
    X, Y = X.view(X.size(0), -1), Y.view(Y.size(0), -1)
    batch_size, dim = X.size()
    t = torch.randn(num_freqs, dim, device=X.device) * sigma
    X_reshaped = X.view(batch_size, dim)
    tX = torch.matmul(t, X_reshaped.t())
    cos_tX = (torch.cos(tX) * wX).mean(1)
    sin_tX = (torch.sin(tX) * wX).mean(1)
    Y_reshaped = Y.view(batch_size, dim)
    tY = torch.matmul(t, Y_reshaped.t())
    cos_tY = (torch.cos(tY) * wY).mean(1)
    sin_tY = (torch.sin(tY) * wY).mean(1)
    loss = (cos_tX - cos_tY) ** 2 + (sin_tX - sin_tY) ** 2
    return loss.mean()