import json
import numpy as np
from matplotlib import pyplot as plt
from plotting import plot, polar2cartesian
from Classes import TwoBodySystem, body_decoder, constant_decoder

# leer fichero de planetas
planets_file = r"bodies.json"
with open(planets_file) as file:
    cuerpos = json.loads(file.read()) # https://starlust.org/the-planets-in-order-from-the-sun/

# crear objetos de planetas
# for s in planets:
    # exec(f"{s} = body_decoder(planets['{s}'])")

Tierra = body_decoder(cuerpos['planetas']['Tierra'])
Tierra1 = body_decoder(cuerpos['planetas']['Tierra'])
Tierra1.set_velocity(30*Tierra.vel)
Tierra1.set_position(Tierra.pos/10000000)
Tierra1.set_velocity([0, 800, 0])
Tierra1.set_position([1, 0, 0])
Tierra1.mass = Tierra.mass
#Tierra1.mass = 1
Tierra2 = body_decoder(cuerpos['planetas']['Tierra'])
#Tierra2.mass = 1
Tierra2.set_position([0, 0, 0])
Tierra2.set_velocity([0, 0, 0])
Jupiter = body_decoder(cuerpos['planetas']['Jupiter'])
Venus = body_decoder(cuerpos['planetas']['Venus'])
Sol = body_decoder(cuerpos['Sol'])
constants = constant_decoder(cuerpos["constantes"])
G = constants["G"]["value"]

# Jupiter Earth system
theta0 = 0
thetafin = 6*np.pi
twoBodySystem = TwoBodySystem(G=G, body1=Tierra1, body2=Tierra2, potential='Log')
r0 = twoBodySystem.r_distance
vr_0 = twoBodySystem.vr_module
h0 = twoBodySystem.spec_angular_momemtum_module
mu = twoBodySystem.mu
E = twoBodySystem.spec_mec_energy
excen = twoBodySystem.eccentricity
V_ef = twoBodySystem.Vef

print("r0:", r0)
print("vr_0:", vr_0)
print("h0:", h0)
print("mu:", mu)
print("E:", E)
print("excentricidad:", excen)
print("V_ef", V_ef)
print("E - V_ef=", E-V_ef)

# twoBodySystem.plot_Enery_potential()

# Calcular Newton
orbit = twoBodySystem.calculate_orbit(theta0, thetafin)
r = orbit[0]
theta = orbit[1]

# pintar resultados
# theta = np.linspace(theta0, thetafin, np.size(r))
#plt.plot(r)
#plt.show()
[rx, ry, rz] = polar2cartesian(r, theta, 0)
#plot(rx, ry, rz, title=title, excen=excen, h=h0, mu=mu, theta_fin=thetafin, show=False)
fig2D = plt.figure(1)
title = "r0="+str(r0)+"; v0="+str(h0/r0)
plt.title(title)
twoBodySystem.update_orbit(theta0, thetafin)
plt.plot(twoBodySystem.orbit_1[0], twoBodySystem.orbit_1[1])
plt.plot(twoBodySystem.orbit_1[0][0], twoBodySystem.orbit_1[1][0], 'bo')
plt.plot(twoBodySystem.orbit_2[0], twoBodySystem.orbit_2[1])
plt.plot(twoBodySystem.orbit_2[0][0], twoBodySystem.orbit_2[1][0], 'o', color='orange')
plt.plot(0, 0, 'rx')
plt.text(0, 0, "CG")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
fig2D.savefig("2BLog")
