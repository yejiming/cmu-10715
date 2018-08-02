# -*- coding: utf-8 -*-

import time

import numpy as np
from scipy.linalg import cholesky
from sklearn.preprocessing import StandardScaler


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
            row = [float(i) for i in line.split(" ")]
            cache.append(row)
        except:
            pass
    f.close()

    return data


def rbf(X1, X2, sigma=1.0, h=1.0):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.dot(np.ones((n2, 1)), (X1 ** 2).sum(axis=1).reshape(1, -1)).T + \
        np.dot(np.ones((n1, 1)), (X2 ** 2).sum(axis=1).reshape(1, -1)) - 2.0 * np.dot(X1, X2.T)
    K = sigma * np.exp(-K / (2 * h**2))
    return K


def parameter_search(X_train, y_train):
    hs = np.logspace(-1, 1, 10) * np.linalg.norm(X_train.std(axis=0))
    sigmas = np.logspace(-1, 1, 10) * y_train.std()
    gamma = y_train.std() * 0.01

    best = -float("inf")
    best_params = None
    for h in hs:
        for sigma in sigmas:
            model = GaussianProcess(sigma=sigma, h=h, gamma=gamma)
            model.fit(X_train, y_train)
            log_likelihood = model.log_marginal_likelihood()
            if log_likelihood > best:
                best = log_likelihood
                best_params = (h, sigma)

    return best_params[0], best_params[1], gamma


def mean_squared_error(pred, truth):
    return ((truth - pred) ** 2).mean()


class GaussianProcess(object):

    def __init__(self, sigma=1.0, h=1.0, gamma=1e-10):
        self.sigma = sigma
        self.h = h
        self.gamma = gamma

    def fit(self, X_train, y_train):
        n = X_train.shape[0]

        self.y_train_mean_ = np.mean(y_train)
        self.X_train_ = X_train
        self.y_train_ = y_train - self.y_train_mean_

        K = rbf(X_train, X_train, self.sigma, self.h)
        self.L_ = cholesky(K + np.eye(n) * self.gamma, lower=True)
        self.alpha_ = np.dot(np.linalg.inv(self.L_.T), np.linalg.inv(self.L_).dot(self.y_train_))

    def predict(self, X_test):
        K12 = rbf(self.X_train_, X_test, self.sigma, self.h)
        K22 = rbf(X_test, X_test, self.sigma, self.h)

        y_mean = np.dot(K12.T, self.alpha_) + self.y_train_mean_

        v = np.dot(np.linalg.inv(self.L_), K12)
        y_cov = K22 - np.dot(v.T, v)

        return y_mean, y_cov

    def log_marginal_likelihood(self):
        log_likelihood = -0.5 * np.dot(self.y_train_, self.alpha_) - \
                         np.log(np.diag(self.L_)).sum() - \
                         self.X_train_.shape[0] / 2.0 * np.log(2 * np.pi)
        return log_likelihood


if __name__ == "__main__":
    data = read_data("data/concrete.mat")
    data["XTrain"] = StandardScaler().fit_transform(data["XTrain"])
    data["yTrain"] = data["yTrain"].flatten()

    start = time.time()
    h, sigma, gamma = parameter_search(data["XTrain"], data["yTrain"])
    model = GaussianProcess(sigma=sigma, h=h, gamma=gamma)
    model.fit(data["XTrain"], data["yTrain"])
    pred, _ = model.predict(data["XTrain"])
    end = time.time()

    print "parameter search and train Gaussian Process cost {} seconds".format(end - start)

    print "Gaussian Process mean squared error is {}".format(mean_squared_error(data["yTrain"], pred))
    print "baseline mean squared error is {}".format(mean_squared_error(data["yTrain"], data["yTrain"].mean()))
