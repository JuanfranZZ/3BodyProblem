import numpy as np

# 3D


def polar2cartesian(r, phi, psi):
    x = r * np.cos(phi) * np.cos(psi)
    y = r * np.sin(phi) * np.cos(psi)
    z = r * np.sin(psi)

    return [x, y, z]

# 2D


def pol2car(r, alpha):  # polar to cartesian
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)
    return np.array([x, y])


def car2pol(x, y):  # cartesian to polar
    r = np.sqrt(x**2 + y**2)
    alpha = np.arctan2(y, x)
    return np.array([r, alpha])