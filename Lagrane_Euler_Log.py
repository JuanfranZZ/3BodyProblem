#  solution of equilibrium lagrange-Euler points

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


def f3(x, mu):  # f3 for convention
    return x - (1 - mu) * (x + mu) / (x + mu) ** 2 - mu * (x - 1 + mu) / (x - 1 + mu) ** 2


def LagrangePoints_Log(mu):
    if isinstance(mu, float):
        L3 = (optimize.newton(f3, -mu - 0.01, args=(mu,), maxiter=1000000), 0)
        L1 = (optimize.newton(f3, 0, args=(mu,), maxiter=1000000), 0)
        L2 = (optimize.newton(f3, 1 - mu + 0.01, args=(mu,), maxiter=1000000), 0)
        L4 = (1 / 2 - mu, np.sqrt(3) / 2)
        L5 = (1 / 2 - mu, -np.sqrt(3) / 2)
    else:
        L3 = [(optimize.newton(f3, -xx - 0.01, args=(xx,), maxiter=1000000), 0) for xx in mu]
        L1 = [(optimize.newton(f3, 0, args=(xx,), maxiter=1000000), 0) for xx in mu]
        L2 = [(optimize.newton(f3, 1 - xx + 0.01, args=(xx,), maxiter=1000000), 0) for xx in mu]
        L4 = [(1 / 2 - xx, np.sqrt(3) / 2) for xx in mu]
        L5 = [(1 / 2 - xx, -np.sqrt(3) / 2) for xx in mu]

    result = [L1, L2, L3, L4, L5]
    return result


if __name__ == "__main__":

    mu = np.linspace(0.01, 0.5, 100)

    L1, L2, L3, L4, L5 = LagrangePoints_Log(mu)

    savefig = True
    plot = True

    if savefig:
        fig = plt.figure(1)
        plt.plot(mu, [L[0] for L in L1])
        plt.title(r'$V_{log} L_1(\mu)$')
        plt.grid(True)
        plt.xlabel(r'$\mu$')
        plt.ylabel('x')
        fig.savefig('Log_L1_mu')

        fig = plt.figure(2)
        plt.plot(mu, [L[0] for L in L2])
        plt.title(r'$V_{log} L_2(\mu)$')
        plt.grid(True)
        plt.xlabel(r'$\mu$')
        plt.ylabel('x')
        fig.savefig('Log_L2_mu')

        fig = plt.figure(3)
        plt.plot(mu, [L[0] for L in L3])
        plt.title(r'$V_{log} L_3(\mu)$')
        plt.grid(True)
        plt.xlabel(r'$\mu$')
        plt.ylabel('x')
        fig.savefig('Log_L3_mu')

        fig = plt.figure(4)
        plt.plot(mu, [L[0] for L in L4])
        plt.title(r'$V_{log} L_4(\mu)$')
        plt.grid(True)
        plt.xlabel(r'$\mu$')
        plt.ylabel('x')
        fig.savefig('Log_L4_mu')

        fig = plt.figure(5)
        plt.plot(mu, [L[0] for L in L5])
        plt.title(r'$V_{log} L_5(\mu)$')
        plt.grid(True)
        plt.xlabel(r'$\mu$')
        plt.ylabel('x')
        fig.savefig('Log_L5_mu')

    if plot:
        fig = plt.figure(6)
        # plt.hlines(0.5, -2, 2, colors='k')  # Draw a horizontal line
        plt.xlim(-2, 2)
        plt.ylim(0, 0.5)

        plt.plot([L[0] for L in L1], mu, 'o', ms=2, label='L1')
        plt.plot([L[0] for L in L2], mu, 'o', ms=2, label='L2')
        plt.plot([L[0] for L in L3], mu, 'o', ms=2, label='L3')

        plt.plot(-mu, mu, 'o', color='grey', ms=5, label='m1')
        plt.plot(1 - mu, mu, 'o', color='darkgrey', ms=5, label='m2')

        plt.ylabel(r'$\mu$')
        plt.xlabel('x')
        plt.title('$V_{Log}$ Lagrange-Euler Collinear Points')
        plt.grid(True)

        plt.legend()
        fig.savefig('LagrangeEulerCollinearLog')

    mu = 0.3
    Ls = LagrangePoints_Log(mu)

    if plot:

        fig = plt.figure(7)
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
        plt.title(r'Lagrange Points $V_{Log}$,'+f' $\mu={mu}$')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.show()
        fig.savefig('LagrangeNewtonPoints_LogPotential')
