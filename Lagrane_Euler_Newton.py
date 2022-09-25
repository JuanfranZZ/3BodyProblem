#  solution of equilibrium lagrange-Euler points

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


def f3(x, mu):
    return x + (1 - mu) / ((x + mu) ** 2) + mu / ((x - 1 + mu) ** 2)


def f1(x, mu):
    return x - (1 - mu) / ((x + mu) ** 2) + mu / ((x - 1 + mu) ** 2)


def f2(x, mu):
    return x - (1 - mu) / ((x + mu) ** 2) - mu / ((x - 1 + mu) ** 2)


def LagrangePoints(mu):
    L1 = (optimize.newton(f1, 0, args=(mu,), maxiter=1000000), 0)
    L2 = (optimize.newton(f2, 0, args=(mu,), maxiter=1000000), 0)
    L3 = (optimize.newton(f3, 0, args=(mu,), maxiter=1000000), 0)
    L4 = (1 / 2 - mu, np.sqrt(3) / 2)
    L5 = (1 / 2 - mu, -np.sqrt(3) / 2)
    result = L1, L2, L3, L4, L5
    return result


if __name__ == "__main__":
    mu = np.linspace(0.01, 0.5, 50)

    L3 = np.array([optimize.newton(f3, -xx-0.01, args=(xx,), maxiter=1000000) for xx in mu])
    L2 = np.array([optimize.newton(f2, 1-xx+0.01, args=(xx,), maxiter=1000000) for xx in mu])
    L1 = np.array([optimize.newton(f1, 0, args=(xx,), maxiter=1000000) for xx in mu])

    savefig = True
    plot = True

    if savefig:
        fig = plt.figure(1)
        plt.plot(mu, L3)
        plt.title(r'L3($\mu$)')
        plt.grid(True)
        plt.xlabel(r'$\mu$')
        fig.savefig('Newtonian_L3_mu')

        fig = plt.figure(2)
        plt.plot(mu, L2)
        plt.title(r'L2($\mu$)')
        plt.grid(True)
        plt.xlabel(r'$\mu$')
        fig.savefig('Newtonian_L2_mu')

        fig = plt.figure(3)
        plt.plot(mu, L1)
        plt.title(r'L1($\mu$)')
        plt.grid(True)
        plt.xlabel(r'$\mu$')
        fig.savefig('Newtonian_L1_mu')

    if plot:
        fig = plt.figure(4)
        # plt.hlines(0.5, -2, 2, colors='k')  # Draw a horizontal line
        plt.xlim(-2, 2)
        plt.ylim(0, 0.5)

        plt.plot(L3, mu, 'o', ms=2, label='L3')
        plt.plot(L2, mu, 'o', ms=2, label='L2')
        plt.plot(L1, mu, 'o', ms=2, label='L1')

        plt.plot(-mu, mu, 'o', color='grey', ms=5, label='m1')
        plt.plot(1 - mu, mu, 'o', color='darkgrey', ms=5, label='m2')

        plt.ylabel(r'$\mu$')
        plt.xlabel('x')
        plt.title('Newton Lagrange-Euler Collinear Points')
        plt.grid(True)

        plt.legend()
        fig.savefig('LagrangeEulerCollinearNewton')

    mu = 0.3
    Ls = LagrangePoints(mu)

    if plot:

        fig = plt.figure(5)
        plt.xlim(-2, 2)
        plt.ylim(-1, 1)

        for ii in range(len(Ls)):
            plt.plot(Ls[ii][0], Ls[ii][1], 'o', color='b')
            plt.text(Ls[ii][0] + 0.005, Ls[ii][1] + 0.005, f"$L_{ii + 1}$")

        plt.plot(-mu, 0, 'o', color='grey', ms=(1 - mu) * 10, label='m1')
        plt.text(-mu, 0, "$m_1$")
        plt.plot(1 - mu, 0, 'o', color='darkgrey', ms=mu * 10, label='m2')
        plt.text(1 - mu, 0, "$m_2$")
        plt.plot(0, 0, 'x', color='red')
        plt.text(0, 0, "CG")
        plt.title(f'Lagrange Points $V_N$, $\mu={mu}$')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.show()
        fig.savefig('LagrangeNewtonPoints_NewtonPotential')

