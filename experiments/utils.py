import torch
import os
import subprocess as sp
from mlutils.pt.training import TrainerBatch
from mlutils.callbacks import Callback
from os import path
import numpy as np
import json


class WeightedSum:
    def __init__(self, name, init, postprocessing=None):
        self.name = name
        self.v = init
        self.w = 0
        self.postprocessing = postprocessing

    def add(self, v, w):
        self.v = self.v + v * w
        self.w = self.w + w

    def get(self):
        # No accumulated value
        if self.w == 0:
            return 0
        v = self.v / self.w
        if self.postprocessing is not None:
            v = self.postprocessing(v)
        return v

    def __repr__(self):
        return '{} {:.5f}'.format(self.name, self.get())


class PerplexStatistics:

    def __init__(self):

        def _item(x):
            return x.item()

        def _exp_item(x):
            return torch.exp(x).item()

        self.stat = {
            'ppx': (WeightedSum('ppx', 0, _exp_item), '', ''),
            'ppx_doc': (WeightedSum('ppx_doc', 0, _exp_item), '', ''),
            'loss': (WeightedSum('loss', 0, _item), 'loss', 'doc_count'),
            'loss_rec': (WeightedSum('loss_rec', 0, _item), 'rec_loss', 'doc_count'),
            'kld': (WeightedSum('kld', 0, _item), 'kld', 'doc_count'),
            'penalty': (WeightedSum('penalty', 0, _item), 'penalty', 'doc_count'),
            'penalty_mean': (WeightedSum('penalty_mean', 0, _item), 'penalty_mean', 'doc_count'),
            'penalty_var': (WeightedSum('penalty_var', 0, _item), 'penalty_var', 'doc_count'),
        }

    def add(self, stat):
        """Accumulate statistics."""
        with torch.no_grad():
            data_batch = stat.pop('data')
            weight = {
                'word_count': data_batch.sum(),
                'doc_count': len(data_batch)
            }

            for s, k, w in self.stat.values():
                if s.name == 'ppx_doc':
                    s.add((stat['minus_elbo'] / data_batch.sum(dim=-1)).sum() / weight['doc_count'],
                          weight['doc_count'])
                elif s.name == 'ppx':
                    s.add(stat['minus_elbo'].sum() / weight['word_count'], weight['word_count'])
                else:
                    if k not in stat:  # skip for compatibility of multiple models.
                        continue
                    s.add(stat[k].mean(), weight[w])
        return self

    def description(self, prefix=''):
        return ' | '.join(['{} {:.5f}'.format(prefix + k, v)
                           for k, v in self.get_dict().items()])

    def get_value(self, k):
        """Get the accumulated value."""
        return self.stat[k][0].get()

    def get_dict(self):
        r = {}
        for k in self.stat.keys():
            t = self.stat[k][0].get()
            if t != 0:
                r[k] = t
        return r


class BatchOperation(TrainerBatch):

    def __init__(self, model, optimizer, loss, device, test_sample=1):
        self.model = model
        self.optimizer = optimizer
        self.dst_device = device
        assert loss in ('mean', 'sum'), 'loss should be mean or sum of batch losses.'
        self.loss = loss
        self.test_sample = test_sample

    def train_batch(self, data, train=True):
        self.model.train(train)

        data = self.place_data_to_device(data)

        if train:
            out = self.model(data)
            self.optimizer.zero_grad()
            if self.loss == 'mean':
                out['loss'].mean().backward()
            else:
                out['loss'].sum().backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                out = self.model(data, n_sample=self.test_sample)

        out.update(data=data)
        return out


def torch_detach(x):
    return x.detach().cpu().numpy()


def save_topics(save_path, vocab, topic_prob, topk=100, logger=None):
    """topic_prob: n_topic x vocab_size.
    Assumed that topic_prob[i] is probability distribution for all i.
    """
    if logger:
        logger.info('saving topics to {}'.format(save_path))

    values, indices = torch.topk(topic_prob, k=topk, dim=-1)
    indices = torch_detach(indices)
    values = torch_detach(values)

    topics = []
    for t in indices:
        topics.append(' '.join([vocab.itos[i] for i in t]))
    with open(save_path+'.topics', 'w') as f:
        f.write('\n'.join(topics))

    str_values = []
    for t in values:
        str_values.append(' '.join([str(v) for v in t]))
    with open(save_path+'.values', 'w') as f:
        f.write('\n'.join(str_values))

    torch.save(topic_prob, save_path + '.pt')


def evaluate_topic_coherence(topic_path, ref_corpus_dir, res_path, logger):
    """Evaluating topic coherence at topic_path whose lines are topics top 10 words.
    The evaluation uses the script at scripts/topics-20news.sh
    """

    if not os.path.exists(topic_path):
        logger.warning('topic file {} not exists'.format(topic_path))
        return -1

    v = -1
    try:
        p = sp.run(['bash', 'scripts/topic_coherence.sh', topic_path, ref_corpus_dir, res_path],
                   encoding='utf-8', timeout=20,
                   stdout=sp.PIPE, stderr=sp.DEVNULL)
        v = float(p.stdout.split('\n')[-3].split()[-1])
    except (ValueError, IndexError, TimeoutError):
        logger.warning('error when calculating topic coherence at {}'.format(topic_path))

    return v


def normalize(v2d, eps=1e-12):
    return v2d / (np.linalg.norm(v2d, axis=1, keepdims=True) + eps)


def wetc(e):
    """embedding matrix: N x D where N is the first N words in a topic, D is the embedding dimension."""
    e = normalize(e)
    t = normalize(e.mean(axis=0, keepdims=True))
    return float(e.dot(t.T).mean())


class EvaluationCallback(Callback):

    def __init__(self, base_dir, vocab, topk=10, corpus_dir="", embedding_path="", metric='npmi', every=10):
        """Evaluate topic coherence based on NPMI or WETC.
        For NPMI: args are, vocab, topk, corpus_dir
        For WETC: args are, embedding_path
        """
        super(EvaluationCallback, self).__init__()
        self.base_dir = base_dir
        self.vocab = vocab
        self.topk = topk
        self.corpus_dir = corpus_dir
        self.every = every
        self.cnt = 0
        self.max_tc = 0
        self.last_tc = 0
        if metric == 'wetc':
            assert os.path.exists(embedding_path), 'embedding file does not exists.'
            self.embedding = np.load(embedding_path)
            assert len(self.embedding) == len(vocab)
        metric = metric.lower()
        assert metric in ['npmi', 'wetc']
        self.metric = metric

    def wetc(self, topics):
        topics = torch_detach(topics)
        idx = np.argsort(topics, axis=1)
        tc = [wetc(self.embedding[idx[-self.topk:]]) for idx in idx]
        save_path = path.join(self.base_dir, 'wetc-topic-{}'.format(self.cnt))
        tc_mean = np.mean(tc)
        tc.append('mean {}'.format(tc_mean))
        with open(save_path, 'w') as f:
            json.dump(tc, f)
        return tc_mean

    def npmi(self, topics):
        save_path = path.join(self.base_dir, 'topic-{}'.format(self.cnt))
        save_topics(save_path, self.vocab, topics, self.topk,
                    self.trainer.logger)
        return evaluate_topic_coherence(save_path+'.topics', self.corpus_dir, save_path+'.res', self.trainer.logger)

    def evaluate_topic_coherence(self):
        topics = self.trainer.trainer_batch.model.get_topics()
        assert topics.size(1) == len(self.vocab), 'topics shape error, should be vocab size {}'.format(len(self.vocab))
        if self.metric == 'npmi':
            tc = self.npmi(topics)
        else:
            tc = self.wetc(topics)
        self.max_tc = max(self.max_tc, tc)
        self.last_tc = tc
        self.trainer.summary_writer.add_scalar('topic_coherence', tc, self.cnt)
        self.trainer.logger.info('topic coherence {}'.format(tc))

    def on_epoch_end(self, epoch, logs=None):
        self.cnt += 1
        if epoch % self.every == 0:
            self.evaluate_topic_coherence()

    def on_train_end(self, logs=None):
        self.evaluate_topic_coherence()

    def get_dict(self):
        return {
            'max_topic_coherence': self.max_tc,
            'last_topic_coherence': self.last_tc
        }


def recover_topic_embedding(topic_word_paths, embedding_path, dataset_dir):
    """Evaluate the WETC of topics generated by NPMI metric."""
    from data_utils import read_dataset
    assert isinstance(topic_word_paths, list), 'Multiple paths should be specified.'
    _, _, vocab = read_dataset(dataset_dir)
    embedding = np.load(embedding_path)
    scores = []
    for p in topic_word_paths:
        with open(p) as f:
            r = []
            for line in f:
                idx = [int(vocab.stoi[w]) for w in line.split()]
                r.append(wetc(embedding[idx]))
            scores.append(r)
    return np.array(scores)


def recover_model(exp_path):
    import json
    from mlutils.exp import yaml_load

    config_path = path.join(exp_path, 'trainer.json')
    if path.exists(config_path):
        print('find trainer.json')
        with open(config_path) as f:
            trainer = json.load(f)
            return BatchOperation.from_config(trainer['trainer_batch'])

    config_path = path.join(exp_path, 'trainer.yaml')
    if path.exists(config_path):
        print('find trainer.yaml')
        trainer = yaml_load(config_path)
        return BatchOperation.from_config(trainer['trainer_batch'])

    return None


def visualize_scale(count_path, scale_path, dataset_dir):
    from data_utils import read_dataset
    from mlutils.exp import yaml_load
    from matplotlib import pyplot as plt
    _, _, vocab = read_dataset(dataset_dir)
    scale = np.load(scale_path)
    if len(scale.shape) == 2:
        scale = scale[0]
    count = yaml_load(count_path)
    kv = sorted(count.items(), key=lambda x: -x[1])
    s = scale[[vocab.stoi[w[0]] for w in kv]]
    # scale = np.array([scale[vocab.stoi[w[0]]] for w in kv])
    plt.plot(s)
