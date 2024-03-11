import numpy as np


def ramsey(Omega, Omegap, delta, tw, tp):
    prob1 = Omegap * np.cos(delta * tw / 2) * np.sin(Omegap * tp)
    prob2 = 2 * delta * np.sin(delta * tw / 2) * np.sin(Omegap * tp / 2) ** 2
    return 1 - Omega ** 2 / Omegap ** 4 * (prob1 - prob2) ** 2


def plot_tw():
    N = 1101001000
    Omega = 2 * np.pi
    Omegap = np.pi / 4
    twlist=0
    pass




if __name__ == "__main__":
    pass
