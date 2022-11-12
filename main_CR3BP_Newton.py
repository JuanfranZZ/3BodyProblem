import numpy as np
from Dynamics_3Bodies_CR3BP import ThreeBodySystem
from Bodies import body_decoder, constant_decoder, Body
import json
from Lagrane_Euler_Newton import LagrangePoints_Kepler
from Lagrane_Euler_Log import LagrangePoints_Log

# leer fichero de planetas
planets_file = r"bodies.json"
with open(planets_file) as file:
    cuerpos = json.loads(file.read()) # https://starlust.org/the-planets-in-order-from-the-sun/

# real
G = 6.67408E-20  # Univ. Gravitational Constant [km3 kg-1 s-2]
mEarth = 5.97219E+24  # Mass of the Earth [kg]
mMoon = 7.34767E+22  # Mass of the Moon [kg]
a = 3.844E+5  # Semi-major axis of Earth and Moon [km]
cuerpo1 = Body("Cuerpo1", mass=mEarth, pos=[0, 0, 0])# mass=0.7, pos=[-0.3, 0, 0])
cuerpo2 = Body("Cuerpo2", mass=mMoon, pos=[a, 0, 0]) #mass=0.3, pos=[0.7, 0, 0])
satelite = Body("cubesat", mass=10, pos=[50000, 0, 0], vel=[1.08, 3.18, 0]) #, pos=[0.2, np.sqrt(3)/2+0.001, 0])

# adim
mu = 0.3
tf = 30
delta = np.array([0, 0.001, 0])

# Lagrange Points
Ls_N = LagrangePoints_Kepler(mu)
Ls_L = LagrangePoints_Log(mu)

cuerpo1 = Body("Cuerpo1", mass=1-mu, pos=[-mu, 0, 0])
cuerpo2 = Body("Cuerpo2", mass=mu, pos=[1-mu, 0, 0])
G = 1

for e, L in enumerate(Ls_N):
    pos_satelite = np.array([L[0], L[1], 0]) + delta
    satelite = Body("cubesat", mass=10, pos=pos_satelite)
    CR3BP_N = ThreeBodySystem(G=G, body1=cuerpo1, body2=cuerpo2, body3=satelite, potential="Kepler")
    CR3BP_N.plot_CR3BP(tf, title=f"$V_N$ L_{e+1}, $\mu=${mu}_y", save=True, plot=False)

for e, L in enumerate(Ls_L):
    pos_satelite = np.array([L[0], L[1], 0]) + delta
    satelite = Body("cubesat", mass=10, pos=pos_satelite)
    CR3BP_N = ThreeBodySystem(G=G, body1=cuerpo1, body2=cuerpo2, body3=satelite, potential="Log")
    CR3BP_N.plot_CR3BP(tf, title=f"Log L_{e+1}, $\mu=${mu}_y", save=True, plot=False)

delta = np.array([0.001, 0, 0])

for e, L in enumerate(Ls_N):
    pos_satelite = np.array([L[0], L[1], 0]) + delta
    satelite = Body("cubesat", mass=10, pos=pos_satelite)
    CR3BP_N = ThreeBodySystem(G=G, body1=cuerpo1, body2=cuerpo2, body3=satelite, potential="Kepler")
    CR3BP_N.plot_CR3BP(tf, title=f"$V_N$ L_{e+1}, $\mu=${mu}_x", save=True, plot=False)

for e, L in enumerate(Ls_L):
    pos_satelite = np.array([L[0], L[1], 0]) + delta
    satelite = Body("cubesat", mass=10, pos=pos_satelite)
    CR3BP_N = ThreeBodySystem(G=G, body1=cuerpo1, body2=cuerpo2, body3=satelite, potential="Log")
    CR3BP_N.plot_CR3BP(tf, title=f"Log L_{e+1}, $\mu=${mu}_x", save=True, plot=False)


