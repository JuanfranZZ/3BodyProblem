# Stability of points
import numpy as np
from Lagrane_Euler_Log import LagrangePoints_Log
from matplotlib import pyplot as plt


def VNxx(x, y, mu):
    r1 = np.sqrt((x+mu)**2+y**2)
    r2 = np.sqrt((x-(1-mu))**2+y**2)
    return 1 - (1 - mu) * (r1 ** 2 - 2 * (x + mu) ** 2) / r1 ** 4 - mu * (r2 ** 2 - 2 * (x - 1 + mu) ** 2) / r2 ** 4


def VNyy(x, y, mu):
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    return 1 - (1 - mu) * (r1 ** 2 - 2 * y ** 2) / r1 ** 4 - mu * (r2 ** 2 - 2 * y ** 2) / r2 ** 4


def VNxy(x, y, mu):
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    return 2*((1-mu)*y*(x+mu)/r1**4 + mu*y*(x-1+mu)/r2**4)


def Stability_Lagrange_Points(mu):
    Ls = LagrangePoints_Log(mu)
    stab = np.zeros((len(Ls), 5))
    for ii, L in enumerate(Ls):
        stabxx = VNxx(L[0], L[1], mu)
        stabxy = VNxy(L[0], L[1], mu)
        stabyy = VNyy(L[0], L[1], mu)
        cond1 = (stabxx*stabyy-stabxy**2 > 0)
        cond2 = 4*(stabxx*stabyy-stabxy**2) < (4+stabxx+stabyy)**2
        stab[ii] = [stabxx, stabxy, stabyy, cond1, cond2]

    return stab


def Check_Stability_Lagrange_Points(mu):
    conds_L1 = np.zeros((len(mu), 5))
    conds_L2 = np.zeros((len(mu), 5))
    conds_L3 = np.zeros((len(mu), 5))
    conds_L4 = np.zeros((len(mu), 5))
    conds_L5 = np.zeros((len(mu), 5))
    for e, m in enumerate(mu):
        Ls = LagrangePoints_Log(m)
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

    plot = True
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
    plt.title(r'$Stability Log\ L_1$')
    ax2.set_ylabel('conds')
    plt.xlabel(r"$\mu$")
    lns = a1 + a2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='right')
    plt.savefig("Stability_Log_L1")
    plt.show()

    fig, ax = plt.subplots()
    a1=ax.plot(mu, stabL2[:, :3],
            label=[r"$V_{xx}V_{yy}-V_{xy}^2$", r"$4(V_{xx}V_{yy}-V_{xy}^2)$", r"$(4-V_{xx}-V_{yy})^2$"])
    ax2 = ax.twinx()
    a2=ax2.plot(mu, stabL2[:, 3:], '--', label=["cond1", "cond2"])
    plt.title(r'$Stability Log\ L_2$')
    ax2.set_ylabel('conds')
    plt.xlabel(r"$\mu$")
    lns = a1 + a2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='right')
    plt.savefig("Stability_Log_L2")
    plt.show()

    fig, ax = plt.subplots()
    a1 = ax.plot(mu, stabL3[:, :3],
                 label=[r"$V_{xx}V_{yy}-V_{xy}^2$", r"$4(V_{xx}V_{yy}-V_{xy}^2)$", r"$(4-V_{xx}-V_{yy})^2$"])
    ax2 = ax.twinx()
    a2 = ax2.plot(mu, stabL3[:, 3:], '--', label=["cond1", "cond2"])
    plt.title(r'$Stability Log\ L_3$')
    ax2.set_ylabel('conds')
    plt.xlabel(r"$\mu$")
    lns = a1 + a2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='right')
    plt.savefig("Stability_Log_L3")
    plt.show()

    fig, ax = plt.subplots()
    a1 = ax.plot(mu, stabL4[:, :3],
                 label=[r"$V_{xx}V_{yy}-V_{xy}^2$", r"$4(V_{xx}V_{yy}-V_{xy}^2)$", r"$(4-V_{xx}-V_{yy})^2$"])
    ax2 = ax.twinx()
    a2 = ax2.plot(mu, stabL4[:, 3:], '--', label=["cond1", "cond2"])
    plt.title(r'$Stability Log\ L_4$')
    ax2.set_ylabel('conds')
    plt.xlabel(r"$\mu$")
    lns = a1 + a2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='right')
    plt.savefig("Stability_Log_L4")
    plt.show()

    fig, ax = plt.subplots()
    a1 = ax.plot(mu, stabL5[:, :3],
                 label=[r"$V_{xx}V_{yy}-V_{xy}^2$", r"$4(V_{xx}V_{yy}-V_{xy}^2)$", r"$(4-V_{xx}-V_{yy})^2$"])
    ax2 = ax.twinx()
    a2 = ax2.plot(mu, stabL5[:, 3:], '--', label=["cond1", "cond2"])
    plt.title(r'$Stability Log\ L_5$')
    ax2.set_ylabel('conds')
    plt.xlabel(r"$\mu$")
    lns = a1 + a2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='right')
    plt.savefig("Stability_Log_L5")
    plt.show()

