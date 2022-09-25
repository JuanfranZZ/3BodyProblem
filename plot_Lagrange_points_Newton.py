import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from Lagrane_Euler_Newton import LagrangePoints_Newton


def f(x, y, mu):
    r1 = np.sqrt((x+mu)**2+y**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2)
    sol = (x**2+y**2)/2 + (1-mu)/r1 + mu/r2
    return sol


def fx(x, y, mu):
    r1 = np.sqrt((x+mu)**2+y**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2)
    sol = x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    return sol


def fy(x, y, mu):
    r1 = np.sqrt((x+mu)**2+y**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2)
    sol = y - (1-mu)*y/r1**3 - mu*y/r2**3
    return sol


mu = 0.3
Ls = LagrangePoints_Newton(mu)
n = 100

for e, L in enumerate(Ls):
    x = np.linspace(L[0]-0.1, L[0]+0.1, n)
    y = np.linspace(L[1]-0.1, L[1]+0.1, n)
    ones = np.ones(n)

    xx, yy = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, f(xx, yy, mu), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.plot(L[0], L[1], f(L[0], L[1], mu), 'ro', zorder=10)
    ax.plot(x, ones * L[1], f(x, ones * L[1], mu), zorder=10)
    ax.plot(L[0]*ones, y, f(ones*L[0], y, mu), zorder=10)
    # ax.axes.set_xlim3d(left=L[0]-0.05, right=L[0]+0.05)
    # ax.axes.set_ylim3d(bottom=L[1]-0.05, top=L[0]+0.05)
    plt.title(f"L{e+1}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, n)
xx, yy = np.meshgrid(x, y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, f(xx, yy, mu), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
for e, L in enumerate(Ls):
    ax.plot(L[0], L[1], f(L[0], L[1], mu), 'ro', zorder=10)
    ax.text(L[0], L[1], f(L[0], L[1], mu), s=f"L{e + 1}", backgroundcolor='white')
# ax.plot(x, ones * L[1], f(x, ones * L[1], mu), zorder=10)
# ax.plot(L[0]*ones, y, f(ones*L[0], y, mu), zorder=10)
# ax.axes.set_xlim3d(left=L[0]-0.05, right=L[0]+0.05)
# ax.axes.set_ylim3d(bottom=L[1]-0.05, top=L[0]+0.05)

plt.xlabel('x')
plt.ylabel('y')
plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.contour(xx, yy, fy(xx, yy, mu), levels=[0],  cmap=cm.coolwarm)
ax.contour(xx, yy, fx(xx, yy, mu), levels=[0],  cmap=cm.coolwarm)
for e, L in enumerate(Ls):
    ax.text(L[0], L[1], 0, s=f"L{e + 1}")
plt.xlabel('x')
plt.ylabel('y')
plt.show()