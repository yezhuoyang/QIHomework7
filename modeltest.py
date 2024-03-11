import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear(x, b, a):
    return b * x + a


def quadratic(x, c, b, a):
    return c * x ** 2 + b * x + a


def generate_data():
    slope = 1
    offset = 0.25
    time = np.linspace(0, 10, 10)
    data = np.random.normal(slope * time + offset, 1)
    sigma = 1 * np.random.normal(np.ones(len(data)), 0.01)

    plt.errorbar(time, data, xerr=0, yerr=sigma)
    # plt.plot(data)
    plt.legend()

    plt.savefig("Generate.png")

    plt.show()

    return time, data


def fit_linear(xdata, ydata):
    popt, pcov = curve_fit(linear, xdata, ydata)

    # Fit line for plotting
    fitted_line = linear(xdata, *popt)

    plt.plot(xdata, linear(xdata, *popt), 'g--',
             label='Linear fit bx+a: b=%5.3f, a=%5.3f' % tuple(popt))
    plt.scatter(xdata, ydata)
    plt.legend()
    plt.savefig("LinearFit.png")
    plt.show()
    print(popt)

    # Calculate residuals
    residuals = ydata - fitted_line

    residual_std = np.std(residuals, ddof=len(popt))  # ddof=len(popt) to correct for the parameters estimated

    return popt, fitted_line, residual_std


def fit_quadratic(xdata, ydata):
    popt, pcov = curve_fit(quadratic, xdata, ydata)

    # Fit line for plotting
    fitted_line = quadratic(xdata, *popt)

    plt.plot(xdata, quadratic(xdata, *popt), 'g--',
             label='Quadratic fit cx^2+bx+a: c=%5.3f, b=%5.3f, a=%5.3f' % tuple(popt))
    plt.scatter(xdata, ydata)

    plt.legend()
    plt.savefig("QuadraticFit.png")
    plt.show()
    print(popt)

    # Calculate residuals
    residuals = ydata - fitted_line

    residual_std = np.std(residuals, ddof=len(popt))  # ddof=len(popt) to correct for the parameters estimated

    return popt, fitted_line, residual_std


def calculate_chi(xdata, fittedxdata, uncertainty):
    '''
    :param xdata: Original xdata
    :param fittedxdata: The xdata calculated from fitted parameters
    '''
    Q = ((xdata - fittedxdata) ** 2 / uncertainty)
    return np.sum(Q)


if __name__ == "__main__":
    time, data = generate_data()

    N = len(data)

    popt, fitteddata, sigma = fit_linear(time, data)
    # fit_linear(time, data)
    # fit_quadratic(time, data)

    sum = calculate_chi(data, fitteddata, sigma)

    print("Linear fitting X^2 test:")
    print("N-k: {}".format(N - 3))

    print("X^2 result:{}".format(sum))

    popt, fitteddata, sigma = fit_quadratic(time, data)
    # fit_linear(time, data)
    # fit_quadratic(time, data)

    sum = calculate_chi(data, fitteddata, sigma)

    print("Quadratic fitting X^2 test:")
    print("N-k: {}".format(N - 3))
    print("X^2 result:{}".format(sum))


    print(sum)
