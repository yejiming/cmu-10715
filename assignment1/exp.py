import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


def calc_mse(pred, truth):
    return np.square(np.array(pred) - truth).mean()


def question_1_3():
    n = 20
    lambd = 0.2
    beta = 100

    dist = stats.expon(scale=1.0/lambd)
    alphas = range(1, 31)

    MLE_scores, MAP_scores = [], []
    for alpha in alphas:
        MLE, MAP = [], []
        for i in range(50):
            samples = np.array([dist.rvs() for _ in range(n)])
            MLE.append(1.0 / samples.mean())
            MAP.append((n + alpha - 1) / (beta + samples.sum()))
        MLE_scores.append(calc_mse(MLE, lambd))
        MAP_scores.append(calc_mse(MAP, lambd))

    plt.semilogy(alphas, MLE_scores, label="MLE")
    plt.semilogy(alphas, MAP_scores, label="MAP")
    plt.ylim(0, 0.1)
    plt.legend(loc="upper right")
    plt.show()


def question_1_4():
    lambd = 0.2
    alpha = 30
    beta = 100

    dist = stats.expon(scale=1.0/lambd)
    ns = [10 * i for i in range(1, 101)]

    MLE_scores, MAP_scores = [], []
    for n in ns:
        MLE, MAP = [], []
        for i in range(50):
            samples = np.array([dist.rvs() for _ in range(n)])
            MLE.append(1.0 / samples.mean())
            MAP.append((n + alpha - 1) / (beta + samples.sum()))
        MLE_scores.append(calc_mse(MLE, lambd))
        MAP_scores.append(calc_mse(MAP, lambd))

    plt.semilogy(ns, MLE_scores, label="MLE")
    plt.semilogy(ns, MAP_scores, label="MAP")
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    question_1_4()
