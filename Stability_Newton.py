# Stability of points
import numpy as np
from Lagrane_Euler_Newton import LagrangePoints_Newton
from matplotlib import pyplot as plt


def A(x, y, mu):
    mu1 = 1 - mu
    mu2 = mu
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    sol = mu1 / (r1 ** 3) + mu2 / (r2 ** 3)

    return sol


def B(x, y, mu):
    mu1 = 1 - mu
    mu2 = mu
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    sol = 3 * (mu1 / (r1 ** 5) + mu2 / (r2 ** 5)) * y ** 2

    return sol


def C(x, y, mu):
    mu1 = 1 - mu
    mu2 = mu
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    sol = 3 * (mu1 * (x + mu2) / (r1 ** 5) + mu2 * (x - mu1) / (r2 ** 5)) * y

    return sol


def D(x, y, mu):
    mu1 = 1 - mu
    mu2 = mu
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    sol = 3 * (mu1 * (x + mu2) ** 2 / (r1 ** 5) + mu2 * (x - mu1) ** 2 / (r2 ** 5))

    return sol


def VNxx(x, y, mu):
    return 1 - A(x, y, mu) + D(x, y, mu)


def VNyy(x, y, mu):
    return 1 - A(x, y, mu) + B(x, y, mu)


def VNxy(x, y, mu):
    return C(x, y, mu)


def Stability_Lagrange_Points(mu):
    Ls = LagrangePoints_Newton(mu)
    stab = np.zeros((len(Ls), 5))
    for ii, L in enumerate(Ls):
        stabxx = VNxx(L[0], L[1], mu)
        stabxy = VNxy(L[0], L[1], mu)
        stabyy = VNyy(L[0], L[1], mu)
        cond1 = (stabxx*stabyy-stabxy**2 > 0)
        cond2 = 4*(stabxx*stabyy-stabxy**2) < (4-stabxx-stabyy)**2
        stab[ii] = [stabxx, stabxy, stabyy, cond1 , cond2]

    return stab


def Check_Stability_Lagrange_Points(mu):
    conds_L1 = np.zeros((len(mu), 5))
    conds_L2 = np.zeros((len(mu), 5))
    conds_L3 = np.zeros((len(mu), 5))
    conds_L4 = np.zeros((len(mu), 5))
    conds_L5 = np.zeros((len(mu), 5))
    for e, m in enumerate(mu):
        Ls = LagrangePoints_Newton(m)
        conds_m = np.zeros((len(Ls), 5))
        for ii, L in enumerate(Ls):
            stabxx = VNxx(L[0], L[1], m)
            stabxy = VNxy(L[0], L[1], m)
            stabyy = VNyy(L[0], L[1], m)
            cond1 = (stabxx*stabyy-stabxy**2 > 0)
            cond2 = 4*(stabxx*stabyy-stabxy**2) < (4-stabxx-stabyy)**2
            cond_1 = stabxx*stabyy-stabxy**2
            cond_2 = 4*(stabxx*stabyy-stabxy**2)
            cond_3 = (4-stabxx-stabyy)**2
            conds_m[ii] = [cond_1, cond_2, cond_3, cond1, cond2]

        conds_L1[e] = conds_m[0]
        conds_L2[e] = conds_m[1]
        conds_L3[e] = conds_m[2]
        conds_L4[e] = conds_m[3]
        conds_L5[e] = conds_m[4]

    return conds_L1, conds_L2, conds_L3, conds_L4, conds_L5


if __name__ == "__main__":

    plot = False
    save = True

    sol = Stability_Lagrange_Points(0.02)

    print(sol)

    mu = np.linspace(0.01, 0.5, 200)
    stabL1, stabL2, stabL3, stabL4, stabL5 = Check_Stability_Lagrange_Points(mu)

    fig, ax = plt.subplots()
    a1 = ax.plot(mu, stabL1[:, :3],
                 label=[r"$V_{xx}V_{yy}-V_{xy}^2$", r"$4(V_{xx}V_{yy}-V_{xy}^2)$", r"$(4-V_{xx}-V_{yy})^2$"])
    ax2 = ax.twinx()
    a2 = ax2.plot(mu, stabL1[:, 3:], '--', label=["cond1", "cond2"])
    plt.title(r'Estabilidad $V_{N}\ L_1$')
    ax2.set_ylabel('conds')
    ax.set(xlabel=r"$\mu$")
    lns = a1 + a2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='right')
    ax.grid()
    plt.savefig("Stability_Newton_L1")
    if plot:
        plt.show()

    fig, ax = plt.subplots()
    a1=ax.plot(mu, stabL2[:, :3],
            label=[r"$V_{xx}V_{yy}-V_{xy}^2$", r"$4(V_{xx}V_{yy}-V_{xy}^2)$", r"$(4-V_{xx}-V_{yy})^2$"])
    ax2 = ax.twinx()
    a2=ax2.plot(mu, stabL2[:, 3:], '--', label=["cond1", "cond2"])
    plt.title(r'Estabilidad $V_{N}\ L_2$')
    ax2.set_ylabel('conds')
    ax.set(xlabel=r"$\mu$")
    lns = a1 + a2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='right')
    ax.grid()
    plt.savefig("Stability_Newton_L2")
    if plot:
        plt.show()

    fig, ax = plt.subplots()
    a1 = ax.plot(mu, stabL3[:, :3],
                 label=[r"$V_{xx}V_{yy}-V_{xy}^2$", r"$4(V_{xx}V_{yy}-V_{xy}^2)$", r"$(4-V_{xx}-V_{yy})^2$"])
    ax2 = ax.twinx()
    a2 = ax2.plot(mu, stabL3[:, 3:], '--', label=["cond1", "cond2"])
    plt.title(r'Estabilidad $V_{N}\ L_3$')
    ax2.set_ylabel('conds')
    ax.set(xlabel=r"$\mu$")
    lns = a1 + a2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='right')
    ax.grid()
    plt.savefig("Stability_Newton_L3")
    if plot:
        plt.show()

    fig, ax = plt.subplots()
    a1 = ax.plot(mu, stabL4[:, :3],
                 label=[r"$V_{xx}V_{yy}-V_{xy}^2$", r"$4(V_{xx}V_{yy}-V_{xy}^2)$", r"$(4-V_{xx}-V_{yy})^2$"])
    ax2 = ax.twinx()
    a2 = ax2.plot(mu, stabL4[:, 3:], '--', label=["cond1", "cond2"])
    plt.title(r'Estabilidad $V_{N}\ L_4$')
    ax2.set_ylabel('conds')
    ax.set(xlabel=r"$\mu$")
    lns = a1 + a2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='right')
    ax.grid()
    plt.savefig("Stability_Newton_L4")
    if plot:
        plt.show()

    fig, ax = plt.subplots()
    a1 = ax.plot(mu, stabL5[:, :3],
                 label=[r"$V_{xx}V_{yy}-V_{xy}^2$", r"$4(V_{xx}V_{yy}-V_{xy}^2)$", r"$(4-V_{xx}-V_{yy})^2$"])
    ax2 = ax.twinx()
    a2 = ax2.plot(mu, stabL5[:, 3:], '--', label=["cond1", "cond2"])
    plt.title(r'Estabilidad $V_{N}\ L_5$')
    ax2.set_ylabel('conds')
    ax.set(xlabel=r"$\mu$")
    lns = a1 + a2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='right')
    ax.grid()
    plt.savefig("Stability_Newton_L5")
    if plot:
        plt.show()


