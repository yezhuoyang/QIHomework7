from qutip import *
import numpy as np
import matplotlib.pyplot as plt

N = 20

hbar = 1
m = 1
omega = np.pi * 2

a = destroy(N)
adag = create(N)

H0 = hbar * omega * (adag * a + 0.5)

zero_fock = basis(N, 0)

x_op = np.sqrt(hbar / 2 / m / omega) * (a + adag)
p_op = 1j * np.sqrt(hbar * m * omega / 2) * (adag - a)


def U0(t):
    return (-1j * t * H0).expm()


def alpha_state(alpha):
    dp = displace(N, alpha)
    return dp * zero_fock


def state(t, alpha):
    return U0(t) * alpha_state(alpha)


def xp_expectation(t, alpha):
    tmp_state = state(t, alpha)
    return expect(x_op, tmp_state), expect(p_op, tmp_state)


def sigma_xp_expectation(t, alpha):
    tmp_state = state(t, alpha)

    xexp, pexp = xp_expectation(t, alpha)

    xexp2 = expect(x_op ** 2, tmp_state)

    pexp2 = expect(p_op ** 2, tmp_state)

    sigmax = np.sqrt(xexp2 - xexp ** 2)

    sigmap = np.sqrt(pexp2 - pexp ** 2)

    return sigmax, sigmap


def plot_xp_phase_diag():
    tlist = np.linspace(0, 2 * np.pi / omega, 1000)
    xlist = []
    plist = []

    for t in tlist:
        xexp, pexp = xp_expectation(t, 1)
        xlist.append(xexp)
        plist.append(pexp)

    plt.plot(xlist, plist, label="XP phase diagram")
    plt.xlabel("Expectation value of X")
    plt.ylabel("Expectation value of P")

    plt.legend()
    plt.show()


def plot_xp_sigma():
    tlist = np.linspace(0, 2 * np.pi / omega, 1000)
    sigmax_list = []
    sigmap_list = []
    sigmaxp_list=[]

    for t in tlist:
        sigmax, sigmap = sigma_xp_expectation(t, alpha=1)
        sigmax_list.append(sigmax)
        sigmap_list.append(sigmap)
        sigmaxp_list.append(sigmax*sigmap)
    plt.figure(1)

    plt.plot(tlist, sigmax_list)
    plt.show()


    plt.figure(2)

    plt.plot(tlist, sigmap_list)
    plt.show()

    plt.figure(3)

    plt.plot(tlist, sigmaxp_list)
    plt.show()



if __name__ == "__main__":
    plot_xp_sigma()
