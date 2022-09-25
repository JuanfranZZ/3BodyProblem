# Inspired in Vallado, Fundamentals in Astrodynamics and applications 4th edition
# Assumptions:
# 1-The mass of the satellite is negligible compared to that
#   of the attracting body. Reasonable for artificial satellites
#   in the foreseeable future.
# 2-The coordinate system chosen is inertial.
# 3-Spherical bodies
# 4-No other forces act on the system except for gravitational
#   forces that act along a line joinning the centers of the two
#   bodies.

import numpy as np
from scipy.integrate import solve_ivp
from scipy import optimize
from matplotlib import pyplot as plt


class Body:
    def __init__(self, name, mass, **kwargs):
        self.name = name
        self.mass = mass
        try:
            if "pos" in kwargs:
                self.set_position(kwargs['pos'])
            else:
                self.pos = None
            if "vel" in kwargs:
                self.set_velocity(kwargs['vel'])
            else:
                self.vel = None
        except Exception as e:
            print(e)
            pass

    def set_position(self, position):
        self.pos = np.array(position)

    def set_velocity(self, velocity):
        self.vel = np.array(velocity)


def body_decoder(obj):
    if "ecc" in obj:
        e = obj["ecc"]
        vel = obj["vel"]*np.sqrt(2/(1-e)-1) #perihelio vel
    else:
        vel = obj["vel"]
    return Body(obj["name"], obj["mass"], pos=obj["pos"], vel=[0, vel, 0])


def unidades_decoder(obj):
    c = {}
    for key, value in obj.items():
        c[str(key)] = value
    return c


def constant_decoder(obj):
    c = {}
    for key, value in obj.items():
        c[str(key)] = value
    return c


class BodySystem:
    def __init__(self, G=None):
        if G:
            self.G = G
        else:
            # self.G = 6.673848*10**(-11) # Nm2/kg2
            self.G = 6.673848*10**(-20)  # km^3/(kg*s^2)
        self.bodies = []

    def add_body(self, body):
        self.bodies.append(body)

    def remove_body(self, body):
        self.bodies.remove(body)






