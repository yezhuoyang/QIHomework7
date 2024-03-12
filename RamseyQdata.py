import numpy as np
import random


def ramsey(Omega, Omegap, delta, tw, tp):
    prob1 = Omegap * np.cos(delta * tw / 2) * np.sin(Omegap * tp)
    prob2 = 2 * delta * np.sin(delta * tw / 2) * np.sin(Omegap * tp / 2) ** 2
    return 1 - Omega ** 2 / Omegap ** 4 * (prob1 - prob2) ** 2


def sample_bernoulli(P):
    return 1 if random.random() < P else 0


def plot_tw(tw):
    N = 11010
    Omega = 2 * np.pi

    delta = np.pi / 4

    Omegap = np.sqrt(delta ** 2 + Omega ** 2)

    '''
    The length of tp is such set to generate a 
    pi/2 pulse 
    '''
    tp = np.pi / 2 / Omegap

    P1 = ramsey(Omega, Omegap, delta, tw, tp)
    oneinstances = 0
    for i in range(0, N):
        print(i)
        sample = sample_bernoulli(P1)
        if sample == 1:
            oneinstances += 1

    return oneinstances / N

import matplotlib.pyplot as plt








if __name__ == "__main__":
    Omega = 2 * np.pi
    tmax = 1000*np.pi / (2 * np.pi)

    tw_list = np.linspace(0, tmax, 1000)

    P1_plot_list = [plot_tw(x) for x in tw_list]

    plt.plot(tw_list, P1_plot_list)

    plt.show()
