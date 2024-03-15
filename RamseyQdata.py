import numpy as np
import random


def ramsey(Omega, Omegap, delta, tw, tp):
    prob1 = Omegap * np.cos(delta * tw / 2) * np.sin(Omegap * tp)
    prob2 = 2 * delta * np.sin(delta * tw / 2) * np.sin(Omegap * tp / 2) ** 2
    return 1 - Omega ** 2 / Omegap ** 4 * (prob1 - prob2) ** 2


def sample_bernoulli(P):
    return 1 if random.random() < P else 0


def plot_tw(tw, N):
    N = N
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


def plot_P1(N):
    Omega = 2 * np.pi
    tmax = 8 * np.pi / Omega

    tw_list = np.linspace(0, tmax, 1000)

    P1_plot_list = [plot_tw(x, N) for x in tw_list]

    plt.plot(tw_list, P1_plot_list, label=r'$P_1$ when N=1')

    plt.xlabel(r'$t_w$')
    plt.ylabel(r'$P_1$')

    plt.title(r'$P_1$ when $N=1$')

    plt.legend()

    plt.savefig("N1P1")
    plt.show()


def find_closest_half(P1_list):
    # This will calculate the absolute difference from 1/2 for each P1 value
    absolute_differences = [abs(P1 - 0.5) for P1 in P1_list]
    # Find the index of the smallest difference
    min_index = absolute_differences.index(min(absolute_differences))
    # Return the index and the P1 value at that index
    return min_index, P1_list[min_index]


def slope_detection(N):
    Omega = 2 * np.pi
    tmax = 8 * np.pi / Omega

    tw_list = np.linspace(0, tmax, 1000)

    P1_plot_list = [plot_tw(x, N) for x in tw_list]

    min_index, value = find_closest_half(P1_plot_list)

    plt.plot(tw_list, P1_plot_list, label=r'$P_1$ when N={}'.format(N))
    plt.axvline(x=tw_list[min_index], color="red", label="tw={}".format(tw_list[min_index]))
    plt.axhline(y=0.5, color="green", label="P1=0.5")
    plt.xlabel(r'$t_w$')
    plt.ylabel(r'$P_1$')

    plt.title(r'$P_1$ when $N={}$'.format(N))

    plt.legend()

    plt.savefig("SlopeFindN{}.png".format(N))
    plt.show()


def plot_detection(N):
    Omega = 2 * np.pi
    tmax = 8 * np.pi / Omega
    delta = np.pi / 4

    tw_list = np.linspace(0, tmax, 1000)

    P1_plot_list = [plot_tw(x, N) for x in tw_list]

    min_index, value = find_closest_half(P1_plot_list)

    phi_list = [delta * tw for tw in tw_list]
    middlephi = phi_list[min_index]

    phi_list = [phi - middlephi for phi in phi_list]

    new_phi_list = []
    new_P1_plot_list = []

    for i in range(0, len(phi_list)):
        if -1 < phi_list[i] < 1:
            new_phi_list.append(phi_list[i])
            new_P1_plot_list.append(P1_plot_list[i])

    theory = [x / 2 + 0.5 for x in new_phi_list]

    plt.plot(new_phi_list, new_P1_plot_list, label=r'$P_1$ when N={}'.format(N))

    plt.plot(new_phi_list, theory, color="red", label=r'$P_1=\frac{{1}}{{2}}\Phi$ when N={}'.format(N))

    plt.xlabel(r'$\Phi$')
    plt.xlim(-1, 1)
    plt.ylabel(r'$P_1$')

    plt.title(r'$P_1$ versus $\Phi$ when $N={}$'.format(N))

    plt.legend()

    plt.savefig("SlopeN{}.png".format(N))
    plt.show()


def plot_uncertainty(N):
    Omega = 2 * np.pi
    tmax = 8 * np.pi / Omega

    delta = np.pi / 4

    Omegap = np.sqrt(delta ** 2 + Omega ** 2)

    '''
    The length of tp is such set to generate a 
    pi/2 pulse 
    '''
    tp = np.pi / 2 / Omegap

    tw_list = np.linspace(0, tmax, 1000)

    P1_plot_list = [plot_tw(x, N) for x in tw_list]

    real_P1_list = [ramsey(Omega, Omegap, delta, tw, tp) for tw in tw_list]

    sigma_P1_list = [np.sqrt(p1 * (1 - p1) / N) for p1 in P1_plot_list]

    plt.plot(tw_list, P1_plot_list, label=r'$P_1$ when N=1')

    plt.xlabel(r'$t_w$')
    plt.ylabel(r'$P_1$')

    plt.title(r'$P_1$ when $N=1$')

    plt.legend()

    plt.savefig("N1P1")
    plt.show()


def plot_uncertainty_chatGPT(N):
    Omega = 2 * np.pi
    tmax = 8 * np.pi / Omega

    delta = np.pi / 4
    Omegap = np.sqrt(delta ** 2 + Omega ** 2)
    tp = np.pi / 2 / Omegap  # pi/2 pulse

    tw_list = np.linspace(0, tmax, 1000)
    P1_plot_list = [plot_tw(x, N) for x in tw_list]
    real_P1_list = [ramsey(Omega, Omegap, delta, tw, tp) for tw in tw_list]
    sigma_P1_list = [np.sqrt(p1 * (1 - p1) / N) for p1 in real_P1_list]

    # Plot the real P1 values
    plt.plot(tw_list, real_P1_list, color="red",label=f'Real $P_1$ when N={N}')

    # Fill the area between P1 + sigma and P1 - sigma
    plt.fill_between(tw_list,
                     [p1 + sigma for p1, sigma in zip(real_P1_list, sigma_P1_list)],
                     [p1 - sigma for p1, sigma in zip(real_P1_list, sigma_P1_list)],
                     color='gray', alpha=0.5, label='Standard Deviation')

    plt.scatter(tw_list, P1_plot_list, label=r'$P_1$ when N=100', s=2)

    plt.xlabel(r'$t_w$')
    plt.ylabel(r'$P_1$')
    plt.title(f'Real $P_1$ with Uncertainty when N={N}')
    plt.legend()
    plt.savefig(f"P1_Uncertainty_N{N}.png")
    plt.show()


import matplotlib.pyplot as plt

if __name__ == "__main__":
    # slope_detection(10)
    # plot_detection(1)
    plot_uncertainty_chatGPT(100)
