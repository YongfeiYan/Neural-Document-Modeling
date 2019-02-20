import os
from os import path
import json
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import OrderedDict
import numpy as np
import re
from string import punctuation

from data_utils import read_pre_embedding


def tfidf_20newsgroup(save_path):
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(norm=None)
    vectorizer.fit(train.data)
    vocab = vectorizer.vocabulary_

    arr = list(vocab.keys())
    varr = vectorizer.transform(arr)
    tfidf = {k: varr[i, vocab[k]] for i, k in enumerate(arr)}
    cnt = super(TfidfVectorizer, vectorizer).transform(train.data).sum(axis=0)
    cnt = {k: cnt[0, vocab[k]] for k in arr}

    with open(save_path, 'w') as f:
        json.dump({'tfidf': tfidf, 'tf': cnt}, f)


def read_stop_words(file_path):
    with open(file_path) as f:
        words = [line.strip() for line in f if line.strip()]
    return set(words)


def save_dataset(save_dir, corpus, vocab):
    """corpus: n x all_vocab, vocab: dict of vocab trimmed(subset of all_vocab)."""
    train, test = corpus
    new_vocab = OrderedDict()
    for k in vocab.keys():
        new_vocab[k] = len(new_vocab) + 1
    itos = {v: k for k, v in vocab.items()}

    def _bow(data):
        bow = {}  # n:f
        wf = {}   # word word word
        for i, j in zip(*data.nonzero()):
            if i not in bow:
                bow[i] = []
            if i not in wf:
                wf[i] = []
            if j not in itos:
                continue
            f = int(data[i, j])
            w = itos[j]
            wf[i].extend([w] * f)
            bow[i].append('{}:{}'.format(new_vocab[w], f))
        bow = [' '.join(v) for v in bow.values() if len(v) > 0]
        wf = [' '.join(v) for v in wf.values() if len(v) > 0]
        return bow, wf

    train_bow, train_txt = _bow(train)
    test_bow, test_txt = _bow(test)

    # save data
    os.makedirs(path.join(save_dir, 'corpus'), exist_ok=True)

    def _write_lines(dst, lines):
        with open(dst, 'w') as f:
            for line in lines:
                f.write('{}\n'.format(line))

    _write_lines(path.join(save_dir, 'corpus/train.txt'), train_txt)
    _write_lines(path.join(save_dir, 'corpus/test.txt'), test_txt)
    _write_lines(path.join(save_dir, 'train.feat'), ['1 {}'.format(line) for line in train_bow])
    _write_lines(path.join(save_dir, 'test.feat'), ['1 {}'.format(line) for line in test_bow])
    _write_lines(path.join(save_dir, 'vocab'), ['{} {}'.format(k, v) for k, v in new_vocab.items()])

    return new_vocab


def should_filter_word(w):
    REMOVE = r'[a-z]+'
    return re.fullmatch(REMOVE, w) is None


def create_20newsgroup(stop_words_path, n_vocab, save_dir, embedding_path,
                       remove=('headers', 'footers', 'quotes')):
    os.makedirs(save_dir, exist_ok=True)
    if path.exists(stop_words_path):
        stopwords = list(read_stop_words(stop_words_path))
    else:
        stopwords = []
    print('size of stopwords', len(stopwords))
    data = fetch_20newsgroups(subset='all', remove=remove)
    counter = CountVectorizer(stop_words=stopwords)
    counter.fit(data.data)
    vocab = counter.vocabulary_
    cnt = counter.transform(data.data).sum(axis=0)
    cnt = sorted([(k, cnt[0, vocab[k]]) for k in vocab.keys() if not should_filter_word(k)], key=lambda x: x[1])

    v = {item[0]: vocab[item[0]] for item in cnt[-n_vocab:]}

    train = fetch_20newsgroups(subset='train', remove=remove)
    train = counter.transform(train.data)
    test = fetch_20newsgroups(subset='test', remove=remove)
    test = counter.transform(test.data)
    new_vocab = save_dataset(save_dir, [train, test], v)
    stoi = {k: v-1 for k, v in new_vocab.items()}
    embedding = read_pre_embedding(embedding_path, stoi, dim=50, skip_lines=0, sep=' ',
                                   unk_init=lambda x: np.random.random(x.shape))
    np.save(path.join(save_dir, 'embedding.npy'), embedding)


if __name__ == '__main__':
    # create 20news dataset
    stop_words_path = 'data/stopwords/stop_words_long.txt'
    n_vocab = 4000
    save_dir = 'data/20news-{}'.format(n_vocab)
    embedding_path = '../paper-analysis/temporal-topic-model/data/glove.6B/glove.6B.50d.txt'
    create_20newsgroup(stop_words_path, n_vocab, save_dir, embedding_path)

    print('finished')
