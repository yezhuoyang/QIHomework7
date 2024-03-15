from qutip import *
import numpy as np
import matplotlib.pyplot as plt

N = 50

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
    return (-1j * t * H0 / hbar).expm()


def alpha_state(alpha):
    dp = displace(N, alpha)
    return dp * zero_fock


def alpha_state_squeeze(alpha1, alpha2):
    sq = squeeze(N, alpha1)
    dp = displace(N, alpha2)
    return dp * sq * zero_fock


def state(t, alpha):
    return U0(t) * alpha_state(alpha)


def state_squeeze(t, alpha1, alpha2):
    return U0(t) * alpha_state_squeeze(alpha1, alpha2)


def xp_expectation(t, alpha):
    tmp_state = state(t, alpha)
    return expect(x_op, tmp_state), expect(p_op, tmp_state)


def xp_expectation_squeeze(t, alpha1, alpha2):
    tmp_state = state_squeeze(t, alpha1, alpha2)
    return expect(x_op, tmp_state), expect(p_op, tmp_state)


def sigma_xp_expectation(t, alpha):
    '''
    Calculate the U(t)|a>, U(t)=e^{-itH} and |a> = D(a) |0>
    '''
    dp = displace(N, alpha)
    zero_fock = basis(N, 0)
    Ut = (-1j * t * H0 / hbar).expm()
    tmp_state = Ut * dp * zero_fock
    '''
    Calculate <X> and <P>
    '''
    xexp = expect(x_op, tmp_state)
    pexp = expect(p_op, tmp_state)
    '''
    Calculate <X^2> and <P^2>
    '''
    xexp2 = expect(x_op ** 2, tmp_state)
    pexp2 = expect(p_op ** 2, tmp_state)
    '''
    Calculate sigmaX and sigmaP
    '''
    sigmax = np.sqrt(xexp2 - xexp ** 2)
    sigmap = np.sqrt(pexp2 - pexp ** 2)

    return sigmax, sigmap


def sigma_xp_expectation_squeeze(t, alpha1, alpha2):
    '''
    Calculate the U(t)|a>, U(t)=e^{-itH} and |a> = D(a) |0>
    '''
    dp = displace(N, alpha1)

    sq = squeeze(N, alpha2)
    zero_fock = basis(N, 0)

    Ut = (-1j * t * H0 / hbar).expm()
    tmp_state = Ut * dp * sq * zero_fock
    '''
    Calculate <X> and <P>
    '''
    xexp = expect(x_op, tmp_state)
    pexp = expect(p_op, tmp_state)
    '''
    Calculate <X^2> and <P^2>
    '''
    xexp2 = expect(x_op ** 2, tmp_state)
    pexp2 = expect(p_op ** 2, tmp_state)
    '''
    Calculate sigmaX and sigmaP
    '''
    sigmax = np.sqrt(xexp2 - xexp ** 2)
    sigmap = np.sqrt(pexp2 - pexp ** 2)

    return sigmax, sigmap


def plot_xp_phase_diag():
    tlist = np.linspace(0, 2 * np.pi / omega, 500)
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
    plt.savefig("XPphase.png")
    plt.show()


def plot_xp_sigma():
    tlist = np.linspace(0, 3 * np.pi / omega, 1000)
    sigmax_list = []
    sigmap_list = []
    sigmaxp_list = []

    for t in tlist:
        sigmax, sigmap = sigma_xp_expectation(t, alpha=1)
        sigmax_list.append(sigmax)
        sigmap_list.append(sigmap)
        sigmaxp_list.append(sigmax * sigmap)
    plt.figure(1)

    plt.plot(tlist, sigmax_list, label=r'$\sigma_x$')
    plt.xlabel("Time")
    plt.title(r'$\sigma_x$ versue Time for Coherent State')
    plt.legend()
    plt.savefig("SigmaXCoherent")
    plt.show()

    plt.figure(2)

    plt.plot(tlist, sigmap_list, label=r'$\sigma_p$')
    plt.xlabel("Time")
    plt.title(r'$\sigma_p$ versue Time for Coherent State')
    plt.legend()
    plt.savefig("SigmaPCoherent")
    plt.show()

    plt.figure(3)

    plt.plot(tlist, sigmaxp_list, label=r'$\sigma_{x}\sigma_{p}$')
    plt.xlabel("Time")
    plt.title("$\sigma_{x}\sigma_{p}$ versue Time")
    plt.legend()
    plt.savefig("SigmaXPCoherent")
    plt.show()


def plot_xp_phase_squeeze_diag():
    tlist = np.linspace(0, 2 * np.pi / omega, 500)
    xlist = []
    plist = []
    xlist_squeeze_1 = []
    plist_squeeze_1 = []

    xlist_squeeze_2 = []
    plist_squeeze_2 = []

    xlist_squeeze_3 = []
    plist_squeeze_3 = []

    for t in tlist:
        xexp, pexp = xp_expectation(t, 1)
        xexp_squeeze_1, pexp_squeeze_1 = xp_expectation_squeeze(t, 1, 0.8)
        xexp_squeeze_2, pexp_squeeze_2 = xp_expectation_squeeze(t, 1, 1.1)
        xexp_squeeze_3, pexp_squeeze_3 = xp_expectation_squeeze(t, 1, 1.2)

        xlist.append(xexp)
        plist.append(pexp)
        xlist_squeeze_1.append(xexp_squeeze_1)
        plist_squeeze_1.append(pexp_squeeze_1)

        xlist_squeeze_2.append(xexp_squeeze_2)
        plist_squeeze_2.append(pexp_squeeze_2)

        xlist_squeeze_3.append(xexp_squeeze_3)
        plist_squeeze_3.append(pexp_squeeze_3)

    plt.plot(xlist, plist, label="XP phase diagram")
    plt.plot(xlist_squeeze_1, plist_squeeze_1, label="Squeezed diagram when r=0.8")
    plt.plot(xlist_squeeze_2, plist_squeeze_2, label="Squeezed diagram when r=1.1")
    plt.plot(xlist_squeeze_3, plist_squeeze_3, label="Squeezed diagram when r=1.2")

    plt.title("Compare the XP phase diagram of different r")
    plt.xlabel("Expectation value of X")
    plt.ylabel("Expectation value of P")

    plt.legend()
    plt.savefig("XPPhasediffr.png")
    plt.show()


def plot_xp_sigma_squeeze():
    tlist = np.linspace(0, 3 * np.pi / omega, 1000)
    sigmax_list = []
    sigmap_list = []
    sigmaxp_list = []

    for t in tlist:
        sigmax, sigmap = sigma_xp_expectation_squeeze(t, alpha1=1, alpha2=0.8)
        sigmax_list.append(sigmax)
        sigmap_list.append(sigmap)
        sigmaxp_list.append(sigmax * sigmap)
    plt.figure(1)

    plt.plot(tlist, sigmax_list, label=r'$\sigma_x$')
    plt.xlabel("Time")
    plt.title("$\sigma_x$ after squeezing when r=0.8")
    plt.legend()
    plt.savefig("SqueezeSigmax.png")
    plt.show()

    plt.figure(2)

    plt.plot(tlist, sigmap_list, label=r'$\sigma_p$')
    plt.xlabel("Time")
    plt.title("$\sigma_p$ after squeezing when r=0.8")
    plt.legend()
    plt.savefig("SqueezeSigmap.png")
    plt.show()

    plt.figure(3)

    plt.plot(tlist, sigmaxp_list, label=r'$\sigma_x\sigma_p$')
    plt.xlabel("Time")
    plt.axhline(y=1 / 2 * hbar, color='red', label=r'1/2\hbar')
    plt.title("$\sigma_x\sigma_p$ after squeezing when r=0.8")
    plt.legend()
    plt.savefig("SqueezeSigmaxSigmap.png")
    plt.show()


def V_coeff(t, args):
    beta = 1
    return beta * np.sin(2 * omega * t)


def drive():
    times = np.linspace(0.0, 4 * np.pi / omega, 2000)

    init_state = zero_fock

    result = sesolve([H0, [x_op, V_coeff], [x_op ** 2, V_coeff]], init_state, times, [x_op, p_op, x_op ** 2, p_op ** 2])

    xexp_list = result.expect[0]
    pexp_list = result.expect[1]

    sigmax_list = np.sqrt((result.expect[2] - xexp_list ** 2))
    sigmap_list = np.sqrt((result.expect[3] - pexp_list ** 2))

    plt.figure(1)
    plt.plot(result.expect[0], result.expect[1], label="XP phase diagram")
    plt.title("XP phase graph when add a V(t) drive")
    plt.legend()
    plt.savefig("xpPhaseVt.png")
    plt.show()

    plt.figure(2)
    plt.plot(times, sigmax_list, label=r'$\sigma_x$')
    plt.xlabel("Time")
    plt.title(r'Evolution of $\sigma_x$ when add a V(t) drive')
    plt.legend()
    plt.savefig("sigxVt.png")
    plt.show()

    plt.figure(3)
    plt.plot(times, sigmap_list, label=r'$\sigma_p$')
    plt.xlabel("Time")
    plt.title(r'Evolution of $\sigma_p$ when add a V(t) drive')
    plt.legend()
    plt.savefig("sigpVt.png")
    plt.show()

    plt.figure(4)
    plt.plot(sigmax_list, sigmap_list, label=r'$\sigma_x\sigma_p$')
    plt.axhline(hbar / 2, color='red', label=r'$\hbar/2$')
    plt.xlabel("Time")
    plt.title(r'Evolution of $\sigma_x\sigma_p$ when add a V(t) drive')
    plt.legend()
    plt.savefig("sigxsigpVt.png")
    plt.show()


def drive_two_terms():
    times = np.linspace(0.0, 4 * np.pi / omega, 2000)

    init_state = zero_fock

    result = sesolve([H0, [x_op ** 2, V_coeff]], init_state, times, [x_op, p_op, x_op ** 2, p_op ** 2])

    xexp_list = result.expect[0]
    pexp_list = result.expect[1]

    sigmax_list = np.sqrt((result.expect[2] - xexp_list ** 2))
    sigmap_list = np.sqrt((result.expect[3] - pexp_list ** 2))

    plt.figure(1)

    plt.scatter(result.expect[0], result.expect[1], label="XP phase diagram")
    plt.title(r'XP phase graph when add a V(t) drive but set $\beta_1=0$')
    plt.savefig("xpPhaseBeta0Vt.png")
    plt.legend()
    plt.show()


    plt.figure(2)
    plt.plot(times, sigmax_list, label=r'$\sigma_x$')
    plt.xlabel("Time")
    plt.title(r'Evolution of $\sigma_x$ when add a V(t) drive but set $\beta_1=0')
    plt.savefig("sigxBeta0Vt.png")
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(times, sigmap_list, label=r'$\sigma_p$')
    plt.xlabel("Time")
    plt.title(r'Evolution of $\sigma_p$ when add a V(t) drive but set $\beta_1=0')
    plt.savefig("sigpBeta0Vt.png")
    plt.legend()
    plt.show()




    plt.figure(4)
    plt.plot(sigmax_list, sigmap_list, label=r'$\sigma_x\sigma_p$')
    plt.xlabel("Time")
    plt.title(r'Evolution of $\sigma_x\sigma_p$ when add a V(t) drive but set $\beta_1=0$')
    plt.axhline(hbar / 2, color='red', label=r'$\hbar/2$')
    plt.savefig("sigxsigpBeta0Vt.png")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    # plot_xp_sigma()
    # plot_xp_phase_squeeze_diag()
    # plot_xp_sigma()
    # plot_xp_phase_squeeze_diag()
    # plot_xp_sigma_squeeze()
    # plot_xp_phase_squeeze_diag()
    # plot_xp_phase_diag()
    #drive()
    # plot_xp_sigma()

    # drive_two_terms()
    # plot_xp_sigma_squeeze()
    drive_two_terms()