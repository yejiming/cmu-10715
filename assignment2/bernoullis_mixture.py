# -*- coding: utf-8 -*-
import time
import random

import numpy as np
import matplotlib.pyplot as plt


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
            row = [int(i) for i in line.split(" ")]
            cache.append(row)
        except:
            pass
    f.close()

    return data


class BernoullisMixture(object):

    def __init__(self, K, n_rounds=10, alpha1=1e-8, alpha2=1e-8):
        self.K = K
        self.n_rounds = n_rounds
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.p = None
        self.mix_p = None
        self.eta = None

    def _estep(self, X_train):
        """ EM算法的expectqation step
        X_train (np.array): 原始数据，N * D矩阵，N是数据量，D是每条数据的维度
        """
        N, D = X_train.shape

        # 避免过小的float溢出，使用log的trick
        log_eta = np.zeros((N, self.K))
        for i in range(N):
            log_eta[i] = np.log(self.mix_p) + np.log(self.p ** X_train[i]).sum(axis=1) + \
                         np.log((1 - self.p) ** (1 - X_train[i])).sum(axis=1)

        # 恢复原始大小
        max_eta = log_eta.max(axis=1)
        for k in range(self.K):
            log_eta[:, k] -= max_eta
        eta = np.exp(log_eta)

        sum_eta = eta.sum(axis=1)
        for k in range(self.K):
            eta[:, k] /= sum_eta
        return eta

    def _mstep(self, X_train):
        """ EM算法的maximization step
        X_train (np.array): 原始数据，N * D矩阵，N是数据量，D是每条数据的维度
        """
        N, D = X_train.shape

        new_p = np.zeros((self.K, D))
        K = self.p.shape[0]
        for k in range(self.K):
            for i in range(N):
                new_p[k] += self.eta[i][k] * X_train[i]
            new_p[k] = (new_p[k] + self.alpha1) / (self.eta[:, k].sum() + self.alpha1*D)
        self.p = new_p

        self.mix_p = (self.eta.sum(axis=0) + self.alpha2) / (self.eta.sum() + self.alpha2*K)

    def fit(self, X_train):
        N, D = X_train.shape

        # 初始化p和mix_p
        self.mix_p = np.ones(self.K).astype(float) / self.K

        p = np.ones((self.K, D))
        for k in range(self.K):
            for d in range(D):
                p[k][d] = random.random()
            p[k] /= np.linalg.norm(p[k])
        self.p = p

        # 使用EM算法更新参数
        for i in range(self.n_rounds):
            self.eta = self._estep(X_train)
            self._mstep(X_train)

    def predict(self, X_test):
        eta = self._estep(X_test)
        return eta.argmax(axis=1)

    def show_clusters(self, rows, cols):
        for k in range(self.K):
            img = self.p[k].reshape((28, 28), order="F")
            plt.subplot(rows, cols, k+1)
            plt.imshow(img, cmap="gray")
            plt.axis("off")

    def show_uniques(self, X_test, y_test):
        clusters = self.predict(X_test)
        uniques = {}
        for k in range(self.K):
            uniques[k] = set(y_test[clusters == k])
        for key in uniques:
            print key, uniques[key]


if __name__ == "__main__":
    data = read_data("data/mnist_binary.mat")
    data["yTrain"] = data["yTrain"].flatten().astype(np.int32)

    start = time.time()
    bm = BernoullisMixture(K=20, n_rounds=20)
    bm.fit(data["XTrain"])
    end = time.time()

    print "training Mixture of Bernoullis cost {} seconds".format(end - start)

    bm.show_uniques(data["XTrain"], data["yTrain"])
    bm.show_clusters(4, 5)

    plt.show()
