import math

import numpy as np
from scipy.special import digamma, polygamma
from matplotlib import pyplot as plt


def read_data():
    f = open("data/estimate.mat")
    data = []
    for line in f:
        try:
            data.append(float(line.strip()))
        except:
            pass
    f.close()
    return np.array(data)


def gradient_descent(data, epoch=100, learning_rate=0.01, init_alpha=1.5):
    n = len(data)
    xbar = data.mean()
    alpha = [init_alpha]
    beta = [init_alpha / xbar]

    for i in range(epoch):
        gradient = n * math.log(alpha[-1]) - n * math.log(xbar) - n * digamma(alpha[-1]) + np.log(data).sum()
        alpha.append(alpha[-1] + learning_rate * gradient)
        beta.append(alpha[-1] / xbar)

    return alpha, beta


def newton_method(data, epoch=100, learning_rate=1, init_alpha=1.5):
    n = len(data)
    xbar = data.mean()
    alpha = [init_alpha]
    beta = [init_alpha / xbar]

    for i in range(epoch):
        prime = n * math.log(alpha[-1]) - n * math.log(xbar) - n * digamma(alpha[-1]) + np.log(data).sum()
        p_prime = n / alpha[-1] - n * polygamma(1, alpha[-1])
        gradient = prime / p_prime
        alpha.append(alpha[-1] - learning_rate * gradient)
        beta.append(alpha[-1] / xbar)

    return alpha, beta


if __name__ == "__main__":
    data = read_data()

    alpha, beta = gradient_descent(data)
    plt.plot(alpha, label="GD")

    alpha, beta = newton_method(data)
    plt.plot(alpha, label="Newton")

    plt.ylim(1, 5.0)
    plt.legend(loc="upper right")

    plt.show()
