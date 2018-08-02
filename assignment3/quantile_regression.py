# -*- coding: utf-8 -*-

import time

import numpy as np
from scipy.optimize import linprog
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
            row = [float(i) for i in line.split(" ")]
            cache.append(row)
        except:
            pass
    f.close()

    return data


class QuantileRegression(object):

    def __init__(self, tau):
        self.tau = tau
        self.beta = None

    def fit(self, X_train, y_train):
        N, D = X_train.shape

        c = np.hstack((np.zeros(D), self.tau*np.ones(N), (1-self.tau)*np.ones(N)))
        Z = np.hstack((X_train, np.eye(N), -np.eye(N)))
        y = y_train
        bounds = [(None, None) for _ in range(D)] + [(0, None) for _ in range(N*2)]

        w = linprog(c, A_eq=Z, b_eq=y, bounds=bounds)
        self.beta = w.x[:D]

    def predict(self, X_test):
        return (X_test * self.beta).flatten()


def show_results(X_train, y_train, tau):
    model = QuantileRegression(tau)
    model.fit(X_train, y_train)
    plt.plot(X_train, model.predict(X_train), label="tau={}".format(tau))
    print "beta value is {} when tau={}".format(model.beta[0], tau)


if __name__ == "__main__":
    data = read_data("data/quantile.mat")
    data["yTrain"] = data["yTrain"].flatten()

    plt.scatter(data["XTrain"], data["yTrain"])

    start = time.time()
    show_results(data["XTrain"], data["yTrain"], 0.25)
    show_results(data["XTrain"], data["yTrain"], 0.50)
    show_results(data["XTrain"], data["yTrain"], 0.75)
    end = time.time()

    print "training Quantile Regression cost {} seconds".format(end - start)

    plt.legend(loc="upper right")
    plt.show()
