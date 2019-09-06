__author__ = 'max'
__maintainer__ = 'takashi'

import torch
from torch.tensor import Tensor


def logsumexp(x: Tensor, dim: int) -> Tensor:
    """
    Args:
        x: A pytorch tensor (any dimension will do)
        dim: int, over which to perform the summation.

    Returns: The result of the log(sum(exp(...))) operation.
    """
    xmax, _ = x.max(dim, keepdim=True)
    xmax_, _ = x.max(dim)
    return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))
