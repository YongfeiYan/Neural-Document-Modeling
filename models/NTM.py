import torch
from torch import nn

from models.utils import kld_normal, topic_covariance_penalty, topic_embedding_weighted_penalty


class NTM(nn.Module):
    def __init__(self, hidden, normal, h_to_z, topics):
        super(NTM, self).__init__()
        self.hidden = hidden
        self.normal = normal
        self.h_to_z = h_to_z
        self.topics = topics

    def forward(self, x, n_sample=1):
        h = self.hidden(x)
        mu, log_sigma = self.normal(h)

        kld = kld_normal(mu, log_sigma)
        rec_loss = 0
        for i in range(n_sample):
            z = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
            z = self.h_to_z(z)
            log_prob = self.topics(z)
            rec_loss = rec_loss - (log_prob * x).sum(dim=-1)
        rec_loss = rec_loss / n_sample

        minus_elbo = rec_loss + kld

        return {
            'loss': minus_elbo,
            'minus_elbo': minus_elbo,
            'rec_loss': rec_loss,
            'kld': kld
        }

    def get_topics(self):
        return self.topics.get_topics()


class GSM(NTM):
    def __init__(self, hidden, normal, h_to_z, topics, penalty):
        # h_to_z will output probabilities over topics
        super(GSM, self).__init__(hidden, normal, h_to_z, topics)
        self.penalty = penalty

    def forward(self, x, n_sample=1):
        stat = super(GSM, self).forward(x, n_sample)
        loss = stat['loss']
        penalty, var, mean = topic_covariance_penalty(self.topics.topic_emb)

        stat.update({
            'loss': loss + penalty * self.penalty,
            'penalty_mean': mean,
            'penalty_var': var,
            'penalty': penalty * self.penalty,
        })

        return stat


class NTMR(NTM):
    def __init__(self, hidden, normal, h_to_z, topics, embedding, penalty):
        super(NTMR, self).__init__(hidden, normal, h_to_z, topics)
        self.penalty = penalty
        self.embedding = embedding

    def forward(self, x, n_sample=1):
        stat = super(NTMR, self).forward(x, n_sample)
        loss = stat['loss']
        penalty = topic_embedding_weighted_penalty(self.embedding.weight,
                                                   self.topics.get_topic_word_logit()) * self.penalty
        stat.update({
            'loss': loss + penalty,
            'penalty': penalty
        })
        return stat
