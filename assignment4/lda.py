# -*- coding: utf-8 -*-

import time
from collections import Counter

import numpy as np


def read_data(path):
    f = open(path)
    data = {}
    cache = []
    key = None

    for i, line in enumerate(f):
        line = line.strip()
        if line.startswith("# name:"):
            key = line.split(" ")[-1]
            cache = []
        if not line:
            data[key] = np.array(cache)
        try:
            if key == "titles":
                if line[0] == "#":
                    continue
                else:
                    row = [" ".join(line.strip().split(" ")[1:])]
            elif key == "vocab":
                if line[0] == "#":
                    continue
                else:
                    row = [line.strip()]
            else:
                row = [int(i) for i in line.split(" ")]
            cache.append(row)
        except:
            pass
    f.close()

    return data


def get_X(vocab, w, d):
    X = np.zeros((d.max() + 1, len(vocab))).astype(int)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = np.all(np.vstack((d == i, w == j)).T, axis=1).sum()
    return X


def display_top_words(w, z, T, vocab):
    for t in range(T):
        idx = Counter(w[z == t]).keys()[:5]
        words = [vocab[i] for i in idx]
        print words


class LDA(object):

    def __init__(self, T, n_iters, alpha, beta):
        self.T = T
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta

        self.word_topic_counts = None

    def _counts(self, w, d, z):
        V = w.max() + 1
        m = d.max() + 1
        T = self.T

        word_topic_counts = np.zeros((V, T))
        for i in range(V):
            for j in range(T):
                word_topic_counts[i][j] = np.all(np.vstack((w == i, z == j)).T, axis=1).sum()

        document_topic_counts = np.zeros((m, T))
        for i in range(m):
            for j in range(T):
                document_topic_counts[i][j] = np.all(np.vstack((d == i, z == j)).T, axis=1).sum()

        return word_topic_counts, document_topic_counts

    def _cond_T(self, w_i, d_i, word_topic_counts_i, document_topic_counts_i):
        p = (word_topic_counts_i[w_i] + self.beta) / (word_topic_counts_i.mean(axis=0) + self.beta) * \
            (document_topic_counts_i[d_i] + self.alpha) / (document_topic_counts_i[d_i].mean() + self.alpha)
        return p

    def _inference(self, w_i):
        return (self.word_topic_counts[w_i] + self.beta) / (self.word_topic_counts.mean(axis=0) + self.beta)

    def gibbs_sampling(self, w, d, z):
        word_topic_counts, document_topic_counts = self._counts(w, d, z)

        for i in range(self.n_iters):
            print i
            for i, (w_i, d_i) in enumerate(zip(w, d)):
                word_topic_counts[w_i][z[i]] -= 1
                document_topic_counts[d_i][z[i]] -= 1
                p_cond = self._cond_T(w_i, d_i, word_topic_counts, document_topic_counts)
                z[i] = p_cond.argmax()
                word_topic_counts[w_i][z[i]] += 1
                document_topic_counts[d_i][z[i]] += 1

        self.word_topic_counts = word_topic_counts
        return z

    def fit(self, X):
        w, d = [], []
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                w += [j for _ in range(X[i][j])]
                d += [i for _ in range(X[i][j])]
        w = np.array(w)
        d = np.array(d)
        z = np.random.randint(0, self.T, w.shape[0])
        self.gibbs_sampling(w, d, z)

    def transform(self, X):
        result = np.zeros((X.shape[0], self.T))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                result[i] += X[i][j] * self._inference(j)
        result /= result.sum(axis=1)[:, np.newaxis]
        return result


if __name__ == "__main__":
    data = read_data("data/reuters.mat")

    T = 20
    n_iters = 100

    vocab = data["vocab"].flatten()
    w = data["w"].astype(int).flatten() - 1
    d = data["d"].astype(int).flatten() - 1
    z = np.random.randint(0, T, w.shape[0])

    start = time.time()
    lda = LDA(T, n_iters, 1., 1.)
    z = lda.gibbs_sampling(w, d, z)
    end = time.time()

    print "LDA gibbs sampling cost {} seconds".format(end - start)

    display_top_words(w, z, T, vocab)
