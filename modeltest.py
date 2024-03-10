import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear(x, a, b):
    return b * x + a


def quadratic(x, a, b, c):
    return c * x ** 2 + b * x + a


def generate_data():
    slope = 1
    offset = 0.25
    time = np.linspace(0, 10, 10)
    data = np.random.normal(slope * time + offset, 1)
    sigma = 1 * np.random.normal(np.ones(len(data)), 0.01)

    plt.errorbar(time, data, xerr=0, yerr=sigma)
    # plt.plot(data)
    plt.show()

    return time, data


def fit_linear(xdata, ydata):
    popt, pcov = curve_fit(linear, xdata, ydata)

    plt.plot(xdata, linear(xdata, *popt), 'g--',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    plt.scatter(xdata, ydata)
    plt.show()
    print(popt)


def fit_quadratic(xdata, ydata):
    popt, pcov = curve_fit(quadratic, xdata, ydata)

    plt.plot(xdata, quadratic(xdata, *popt), 'g--',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.scatter(xdata, ydata)
    plt.show()
    print(popt)


def calculate_chi(xdata, mudata, sigma):
    '''
    :param xdata: Original xdata
    :param mudata: The average value of x
    :param sigma: The uncertainty of x
    :return: X^2 of this dataset
    '''
    Q = ((xdata - mudata) / sigma) ** 2
    return np.sum(Q)


if __name__ == "__main__":
    time, data = generate_data()
    # fit_linear(time, data)
    fit_quadratic(time, data)
