from os import path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import copy


class DocDataset(Dataset):

    def __init__(self, docs, n_vocab, device):
        """docs are sparse BoW representation."""
        super(DocDataset, self).__init__()
        self.docs = docs
        self.n_vocab = n_vocab
        self.device = device

    def __getitem__(self, item):
        """Return float32 Bow representation."""
        d = self.docs[item]

        v = np.zeros(self.n_vocab, dtype=np.float32)
        for w, f in d:
            v[w] += f
        return torch.Tensor(v).to(self.device)

    def __len__(self):
        return len(self.docs)


class Vocab:
    def __init__(self, itos):
        self.itos = copy.copy(itos)
        self.stoi = {v: k for k, v in enumerate(itos)}

    def __call__(self, s):
        return self.stoi[s]

    def __len__(self):
        return len(self.itos)


def read_docs(file):
    docs = []
    print('reading', file)
    with open(file) as f:
        for line in f:
            doc = [t.split(':') for t in line.split()[1:]]
            doc = [(int(t)-1, float(f)) for t, f in doc if float(f) > 0]
            if len(doc) > 0:
                docs.append(doc)
            else:
                print('null doc!')
    return docs


def read_dataset(data_dir):
    """Read prefix+train.feat prefix+test.feat prefix+vocab."""
    train_docs = read_docs(path.join(data_dir, 'train.feat'))
    test_docs = read_docs(path.join(data_dir, 'test.feat'))

    print('creating dictionary')
    id2word = []
    with open(path.join(data_dir, 'vocab')) as f:
        id2word.extend([line.strip().split(' ')[0] for line in f])
    dictionary = Vocab(id2word)

    return train_docs, test_docs, dictionary


def load_data(data_dir, batch_size, dev_ratio, device):
    train_docs, test_docs, vocab = read_dataset(data_dir)

    if dev_ratio > 0:
        print('splitting train, dev datasets')
        train_docs, dev_docs = train_test_split(train_docs, test_size=dev_ratio, shuffle=True)
        print('train, dev, test', len(train_docs), len(dev_docs), len(test_docs))

        train_loader = DataLoader(DocDataset(train_docs, len(vocab), device), batch_size, drop_last=False,
                                  num_workers=0)
        dev_loader = DataLoader(DocDataset(dev_docs, len(vocab), device), batch_size, drop_last=False,
                                num_workers=0)
        test_loader = DataLoader(DocDataset(test_docs, len(vocab), device), batch_size, drop_last=False,
                                 num_workers=0)

        return train_loader, dev_loader, test_loader, vocab

    else:
        print('train, test', len(train_docs), len(test_docs))

        train_loader = DataLoader(DocDataset(train_docs, len(vocab), device), batch_size, drop_last=False,
                                  num_workers=0)
        test_loader = DataLoader(DocDataset(test_docs, len(vocab), device), batch_size, drop_last=False,
                                 num_workers=0)

        return train_loader, test_loader, test_loader, vocab


def read_pre_embedding(emb_path, stoi, num_embedding=None, dim=None, sep=' ', skip_lines=0, unk_init=np.zeros_like):
    """Read pre trained embedding from files.
    emb_path: first line contains total lines and embedding dimension. other should be specified in dim.
    file format:
        total_lines dim , optimal
        w embedding
        ...

    num_embedding: optional, default to len(stoi).
    """
    if num_embedding is None:
        num_embedding = len(stoi)
    with open(emb_path) as f:
        if dim is None:
            assert not skip_lines, 'dim should be specified in the first line'
            rows, dim = f.readline().split()
            dim = int(dim)
            print('total rows', rows, 'dim of embedding', dim)
        else:
            print('dim of embedding', dim)
            for _ in range(skip_lines):
                f.readline()

        vectors = np.zeros((num_embedding, dim), dtype=np.float32)
        assigned = [0] * num_embedding

        for line in f:
            line = line.strip('\n').split(sep)
            w = sep.join(line[:-dim])
            idx = stoi.get(w, None)
            if idx is None:
                continue
            vec = np.asarray(list(map(float, line[-dim:])), dtype=np.float32)
            idx = stoi[w]
            assigned[idx] = 1
            vectors[idx] = vec

        tot = sum(assigned)
        print(tot, 'of', len(stoi), ' vectors read, rate:', tot / len(stoi))

        for i, b in enumerate(assigned):
            if not b:
                vectors[i] = unk_init(vectors[i])

    return vectors
