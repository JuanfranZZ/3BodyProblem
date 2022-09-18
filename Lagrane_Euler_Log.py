#  solution of equilibrium lagrange-Euler points

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


def f1(x, mu):
    return x - (1 - mu) * (x+mu) / (x + mu)**2 - mu * (x - 1 + mu) / (x - 1 + mu)**2


def LagrangePoints(mu):
    L1 = (optimize.newton(f1, 0, args=(mu,), maxiter=1000000), 0)
    L2 = (1 / 2 - mu, np.sqrt(3) / 2)
    L3 = (1 / 2 - mu, -np.sqrt(3) / 2)
    result = L1, L2, L3
    return result


if __name__=="__main__":

    mu = np.linspace(0.01, 0.5, 100)

    L1 = np.array([optimize.newton(f1, 0, args=(xx,), maxiter=1000000) for xx in mu])

    savefig = True
    plot = True

    if savefig:
        fig = plt.figure(3)
        plt.plot(mu, L1)
        plt.title(r'L1($\mu$)')
        plt.grid(True)
        plt.xlabel(r'$\mu$')
        fig.savefig('Log_L1_mu')

    if plot:

        fig = plt.figure(4)
        # plt.hlines(0.5, -2, 2, colors='k')  # Draw a horizontal line
        plt.xlim(-2, 2)
        plt.ylim(0, 0.5)

        plt.plot(L1, mu, 'o', ms=2, label='L1')

        # Esto está bien?
        plt.plot(-mu, mu, 'o', color='grey', ms=5, label='m1')
        plt.plot(1-mu, mu, 'o', color='darkgrey', ms=5, label='m2')

        plt.ylabel(r'$\mu$')
        plt.xlabel('x')
        plt.title('Log Lagrange-Euler Collinear Points')
        plt.grid(True)

        plt.legend()
        fig.savefig('LagrangeEulerCollinearLog')
