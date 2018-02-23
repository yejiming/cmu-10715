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


class NaiveBayes(object):

    def __init__(self):
        self.prior = {}
        self.params = {}

    def fit(self, X_train, y_train):
        # get prior distribution
        for key, value in collections.Counter(y_train).items():
            self.prior[key] = float(value) / y_train.size

        # get posterior distribution
        for col in range(X_train.shape[1]):
            for label in set(y_train):
                small_data = X_train[:, col][y_train == label]
                self.params[(col, label)] = (small_data.mean(), small_data.std())

    def predict(self, X_test):
        pred = []
        for obs in X_test:
            max_p = 0.0
            max_label = 0
            for label, prior_p in self.prior.items():
                posterior_p = prior_p
                for i, value in enumerate(obs):
                    mean, std = self.params[(i, label)]
                    posterior_p *= scipy.stats.norm.pdf(value, loc=mean, scale=std)
                if posterior_p > max_p:
                    max_p = posterior_p
                    max_label = label
            pred.append(max_label)
        return pred


if __name__ == "__main__":
    data = read_data("data/iris.mat")
    data["yTrain"] = data["yTrain"].flatten().astype(np.int32)
    data["yTest"] = data["yTest"].flatten().astype(np.int32)

    nb = NaiveBayes()
    nb.fit(data["XTrain"], data["yTrain"])
    pred = nb.predict(data["XTest"])

    print (pred == data["yTest"]).sum() / float(len(pred))
