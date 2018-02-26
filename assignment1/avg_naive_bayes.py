import math
import collections

import numpy as np
import scipy.stats


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


class AvgNaiveBayes(object):

    def __init__(self, beta, eta):
        self.beta = beta
        self.eta = eta
        self._labels = {}
        self._prior = {}
        self._params = {}
        self._log_px = {}
        self._log_cond_px = {}

    def _get_pdf(self, **kwargs):
        if "x" not in kwargs and "y" in kwargs:
            return self._prior[kwargs["y"]]
        if "x" in kwargs and "y" not in kwargs:
            mean, std = self._params[kwargs["col"]]
            return scipy.stats.norm.pdf(kwargs["x"], loc=mean, scale=std)
        if "x" in kwargs and "y" in kwargs:
            mean, std = self._params[(kwargs["col"], kwargs["y"])]
            return scipy.stats.norm.pdf(kwargs["x"], loc=mean, scale=std)
        else:
            raise ValueError("Either x or y need to be input.")

    def fit(self, X_train, y_train):
        self._labels = set(y_train)

        # get prior distribution
        for key, value in collections.Counter(y_train).items():
            self._prior[key] = float(value) / y_train.size

        # get P(x) and P(x|y)
        for col in range(X_train.shape[1]):
            small_data = X_train[:, col]
            self._params[col] = (small_data.mean(), small_data.std())
            for label in set(y_train):
                small_data = X_train[:, col][y_train == label]
                self._params[(col, label)] = (small_data.mean(), small_data.std())

        # get cumulative probabilities in training data
        for obs, y in zip(X_train, y_train):
            for col, x in enumerate(obs):
                self._log_px.setdefault(col, 0.0)
                self._log_cond_px.setdefault(col, 0.0)
                self._log_px[col] += math.log(self._get_pdf(x=x, col=col))
                self._log_cond_px[col] += math.log(max(self._get_pdf(x=x, y=y, col=col), self.eta))

    def predict(self, X_test):
        pred = []
        for obs in X_test:
            max_p = -999999
            max_label = 0
            for y in self._labels:
                log_posterior_p = math.log(self._get_pdf(y=y))
                for col, x in enumerate(obs):
                    _log_px = self._log_px[col] + math.log(max(self._get_pdf(x=x, col=col), self.eta))
                    _log_cond_px = self._log_cond_px[col] + math.log(max(self._get_pdf(x=x, y=y, col=col), self.eta))
                    m = max([_log_px, _log_cond_px])
                    log_posterior_p += (m + math.log(math.exp(_log_px - m) + math.exp(_log_cond_px - m) / self.beta))
                if log_posterior_p > max_p:
                    max_p = log_posterior_p
                    max_label = y
            pred.append(max_label)
        return pred


if __name__ == "__main__":
    data = read_data("data/iris.mat")
    data["yTrain"] = data["yTrain"].flatten().astype(np.int32)
    data["yTest"] = data["yTest"].flatten().astype(np.int32)

    nb = AvgNaiveBayes(beta=1e20, eta=1e-10)
    nb.fit(data["XTrain"], data["yTrain"])
    pred = nb.predict(data["XTest"])

    print (pred == data["yTest"]).sum() / float(len(pred))
