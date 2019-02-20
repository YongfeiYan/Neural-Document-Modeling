import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


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


class Sequential(nn.Sequential):
    """Wrapper for torch.nn.Sequential."""
    def __init__(self, args):
        super(Sequential, self).__init__(args)


def get_mlp(features, activate):
    """features: mlp size of each layer, append activation in each layer except for the first layer."""
    if isinstance(activate, str):
        activate = getattr(nn, activate)
    layers = []
    for in_f, out_f in zip(features[:-1], features[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activate())
    return nn.Sequential(*layers)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        if len(input) == 1:
            return input[0]
        return input


class Topics(nn.Module):
    def __init__(self, k, vocab_size, bias=True):
        super(Topics, self).__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.topic = nn.Linear(k, vocab_size, bias=bias)

    def forward(self, logit):
        # return the log_prob of vocab distribution
        return torch.log_softmax(self.topic(logit), dim=-1)

    def get_topics(self):
        return torch.softmax(self.topic.weight.data.transpose(0, 1), dim=-1)

    def get_topic_word_logit(self):
        """topic x V.
        Return the logits instead of probability distribution
        """
        return self.topic.weight.transpose(0, 1)


class EmbTopic(nn.Module):
    def __init__(self, embedding, k):
        super(EmbTopic, self).__init__()
        self.embedding = embedding
        n_vocab, topic_dim = embedding.weight.size()
        self.k = k
        self.topic_emb = nn.Parameter(torch.Tensor(k, topic_dim))
        self.reset_parameters()

    def forward(self, logit):
        # return the log_prob of vocab distribution
        logit = (logit @ self.topic_emb) @ self.embedding.weight.transpose(0, 1)
        return torch.log_softmax(logit, dim=-1)

    def get_topics(self):
        return torch.softmax(self.topic_emb @ self.embedding.weight.transpose(0, 1), dim=-1)

    def reset_parameters(self):
        init.normal_(self.topic_emb)
        # init.kaiming_uniform_(self.topic_emb, a=math.sqrt(5))
        init.normal_(self.embedding.weight, std=0.01)

    def extra_repr(self):
        k, d = self.topic_emb.size()
        return 'topic_emb: Parameter({}, {})'.format(k, d)


# class ETopic(nn.Module):
#     def __init__(self, embedding, k):
#         super(ETopic, self).__init__()
#         n_vocab, topic_dim = embedding.weight.size()
#         self.embedding = nn.Parameter(torch.Tensor(n_vocab, topic_dim))
#         self.k = k
#         self.topic_emb = nn.Parameter(torch.Tensor(k, topic_dim))
#         self.reset_parameters()
#
#     def forward(self, logit):
#         # return the log_prob of vocab distribution
#         logit = (logit @ self.topic_emb) @ self.embedding.transpose(0, 1)
#         return torch.log_softmax(logit, dim=-1)
#
#     def get_topics(self):
#         return torch.softmax(self.topic_emb @ self.embedding.transpose(0, 1), dim=-1)
#
#     def reset_parameters(self):
#         init.normal_(self.topic_emb)
#         # init.normal_(self.topic_emb, std=0.01)
#         init.normal_(self.embedding, std=0.01)
#
#     def extra_repr(self):
#         k, d = self.topic_emb.size()
#         return 'topic_emb: Parameter({}, {})\nembedding: Parameter({}, {})'.format(k, d, self.embedding.size(0),
#                                                                                    self.embedding.size(1))


class ScaleTopic(nn.Module):
    def __init__(self, k, vocab_size, bias=True, logit_importance=True, s=2):
        super(ScaleTopic, self).__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.importance = nn.Parameter(torch.Tensor(k, vocab_size))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.Tensor(1, vocab_size)))
        self.scale = nn.Parameter(torch.Tensor(1, vocab_size))
        self.s = s
        self.logit_importance = logit_importance
        self.reset_parameters()

    def forward(self, logit):
        scale = torch.sigmoid(self.scale) * self.s
        if self.logit_importance:
            topics = self.importance * scale
        else:
            topics = torch.softmax(self.importance, dim=-1) * scale

        r = logit @ topics
        if hasattr(self, 'bias'):
            r = r + self.bias

        return torch.log_softmax(r, dim=-1)

    def get_topics(self):
        return torch.softmax(self.importance, dim=-1)

    def get_topic_word_logit(self):
        return self.importance

    def reset_parameters(self):
        init.kaiming_uniform_(self.importance, a=math.sqrt(5))
        init.zeros_(self.scale)
        if hasattr(self, 'bias'):
            init.zeros_(self.bias)


def topic_covariance_penalty(topic_emb, EPS=1e-12):
    """topic_emb: T x topic_dim."""
    normalized_topic = topic_emb / (torch.norm(topic_emb, dim=-1, keepdim=True) + EPS)
    cosine = (normalized_topic @ normalized_topic.transpose(0, 1)).abs()
    mean = cosine.mean()
    var = ((cosine - mean) ** 2).mean()
    return mean - var, var, mean


def topic_embedding_weighted_penalty(embedding_weight, topic_word_logit, EPS=1e-12):
    """embedding_weight: V x dim, topic_word_logit: T x V."""
    w = topic_word_logit.transpose(0, 1)  # V x T

    nv = embedding_weight / (torch.norm(embedding_weight, dim=1, keepdim=True) + EPS)  # V x dim
    nw = w / (torch.norm(w, dim=0, keepdim=True) + EPS)  # V x T
    t = nv.transpose(0, 1) @ w  # dim x T
    nt = t / (torch.norm(t, dim=0, keepdim=True) + EPS)  # dim x T
    s = nv @ nt  # V x T
    return -(s * nw).sum()  # minus for minimization
