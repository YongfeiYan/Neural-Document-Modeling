import torch
from torch import nn
from torch.nn import init


def kld_normal(mu, log_sigma):
    """KL divergence to standard normal distribution.
    mu: batch_size x dim
    log_sigma: batch_size x dim
    """
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)


class NormalParameter(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormalParameter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_sigma = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def forward(self, h):
        return self.mu(h), self.log_sigma(h)

    def reset_parameters(self):
        init.zeros_(self.log_sigma.weight)
        init.zeros_(self.log_sigma.bias)


class Topics(nn.Module):
    def __init__(self):
        super(Topics, self).__init__()
        pass

    def forward(self, logit):
        pass
