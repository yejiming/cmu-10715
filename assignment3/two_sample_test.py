# -*- coding: utf-8 -*-

import numpy as np
import scipy.io


def rbf(X1, X2, sigma=1.0, h=1.0):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.dot(np.ones((n2, 1)), (X1 ** 2).sum(axis=1).reshape(1, -1)).T + \
        np.dot(np.ones((n1, 1)), (X2 ** 2).sum(axis=1).reshape(1, -1)) - 2.0 * np.dot(X1, X2.T)
    K = sigma * np.exp(-K / (2 * h**2))
    return K


def calc_MMD(X, Y, sigma, h):
    n = X.shape[0]
    MMD = np.sqrt(2.0 / n - 2.0 / (n * n) * np.diag(rbf(X, Y, sigma, h)).sum())
    return MMD


if __name__ == "__main__":
    data = scipy.io.loadmat("data/data.mat")["data"]
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1].reshape(-1, 1)

    print "MMD is {} when sigma=1.0 and h=0.1".format(calc_MMD(X, Y, 1.0, 0.1))
    print "MMD is {} when sigma=1.0 and h=1.0".format(calc_MMD(X, Y, 1.0, 1.0))
    print "MMD is {} when sigma=1.0 and h=10".format(calc_MMD(X, Y, 1.0, 10.0))
