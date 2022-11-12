#  jacobi constante que define las zonas donde puede estar la partícula

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from Lagrane_Euler_Log import LagrangePoints_Log
from Lagrane_Euler_Newton import LagrangePoints_Kepler


def CJ_Log(x, y, z, mu):
    mu2 = mu
    mu1 = 1 - mu

    r1 = np.sqrt((x + mu2) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - mu1) ** 2 + y ** 2 + z ** 2)

    CJ = x ** 2 + y ** 2 - 2 * (mu1 * np.log(r1) + mu2 * np.log(r2))
    return CJ


def CJ_Kepler(x, y, z, mu):
    mu2 = mu
    mu1 = 1 - mu

    r1 = np.sqrt((x + mu2) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - mu1) ** 2 + y ** 2 + z ** 2)

    CJ = x ** 2 + y ** 2 + 2 * (mu1 / r1 + mu2 / r2)
    return CJ


if __name__ == "__main__":
    mu = 0.3

    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)

    xx, yy = np.meshgrid(x, y)

    option = 'Newton'

    if option == 'Log':
        CJ_general = np.round(CJ_Log(xx, yy, 0, mu), 4)
        Ls = LagrangePoints_Log(mu)
    elif option == 'Newton':
        CJ_general = np.round(CJ_Kepler(xx, yy, 0, mu), 4)
        Ls = LagrangePoints_Kepler(mu)

    Lx, Ly = [], []
    try:
        for (xi, yi) in Ls:
            Lx.append(xi)
            Ly.append(yi)

        Lx = np.array(Lx)
        Ly = np.array(Ly)

    except:
        Lx = Ls[0]
        Ly = Ls[1]

    if option == 'Log':
        CJ_Lagrange = CJ_Log(Lx, Ly, 0, mu)
    elif option == 'Newton':
        CJ_Lagrange = CJ_Kepler(Lx, Ly, 0, mu)

    # Lagrange contours

    fig = plt.figure(1)
    plt.title(f"CJ with $\mu$ = {mu}")
    cLagrange = plt.contour(xx, yy, CJ_general, levels=CJ_Lagrange[-2::-1], cmap=cm.Set1, zorder=-1)
    if option == 'Log':
        c = plt.contour(xx, yy, CJ_general, levels=[1, 2], cmap=cm.ocean)
    elif option == 'Newton':
        c = plt.contour(xx, yy, CJ_general, levels=[3, 5], cmap=cm.ocean)
    plt.clabel(cLagrange, inline=True)
    plt.clabel(c, inline=True)

    # General contours
    plt.contour(xx, yy, CJ_general, levels=30)

    # masses
    s1 = 100 * (1 - mu)
    s2 = 100 * mu

    plt.scatter([-mu], [0], s=s1, color='grey', zorder=1)
    plt.text(-mu - 0.1, -0.15, r'$m_1$')
    plt.scatter([1 - mu], [0], s=s2, color='darkgrey', zorder=1)
    plt.text(1 - mu - 0.1, -0.15, r'$m_2$')

    plt.scatter(Lx, Ly, marker='.', color='b', zorder=1)
    plt.text(Lx[0] - 0.05, Ly[0] - 0.15, r'L$_1$')
    plt.text(Lx[1] + 0.05, Ly[1] - 0.05, r'L$_2$')
    plt.text(Lx[2] - 0.2, Ly[2] - 0.05, r'L$_3$')
    plt.text(Lx[3] - 0.05, Ly[3] + 0.05, r'L$_4$')
    plt.text(Lx[4] - 0.05, Ly[4] - 0.15, r'L$_5$')

    plt.axis('off')
    plt.axis('equal')
    if option == 'Log':
        fig.savefig('CJ_contour_LagrangePoints_Log')
    elif option == 'Newton':
        fig.savefig('CJ_contour_LagrangePoints_Newton')
