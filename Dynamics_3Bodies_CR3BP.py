from Bodies import BodySystem, Body
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np
from Lagrane_Euler_Kepler import LagrangePoints_Kepler
from Lagrane_Euler_Log import LagrangePoints_Log
from Jacobi_Kepler import CJ_Kepler
from Jacobi_Log import CJ_Log


class ThreeBodySystem(BodySystem):
    def __init__(self, G, body1, body2, body3, potential):
        self.potential = potential
        self.bodies = [body1, body2, body3]
        self.r1 = self.bodies[2].pos - self.bodies[0].pos
        self.r2 = self.bodies[2].pos - self.bodies[1].pos
        self.r_distance = np.linalg.norm(self.bodies[1].pos - self.bodies[0].pos)
        self.Mtotal = self.bodies[0].mass + self.bodies[1].mass
        self.mu = self.bodies[1].mass / self.Mtotal
        self.tolerance = 0.001
        self.t_adim = (self.r_distance ** 3 / (G * self.Mtotal)) ** (1 / 2)

    def get_tolerance(self):
        return self.tolerance

    def calculate_orbit(self, tf):
        state = np.zeros(6)
        state[:3] = self.bodies[2].pos / self.r_distance
        state[3:] = self.bodies[2].vel * self.t_adim / self.r_distance
        t = (0, tf)

        def dynamic_CR3BP(t, _state):  # en ejes que rotan con los cuerpos principales
            mu = self.mu
            x = _state[0]
            y = _state[1]
            z = _state[2]
            x_dot = _state[3]
            y_dot = _state[4]
            z_dot = _state[5]
            if self.potential == 'Kepler':
                x_ddot = x + 2 * y_dot - ((1 - mu) * (x + mu)) / ((x + mu) ** 2 + y ** 2 + z ** 2) ** (3 / 2) \
                         - (mu * (x - (1 - mu))) / ((x - (1 - mu)) ** 2 + y ** 2 + z ** 2) ** (3 / 2)
                y_ddot = y - 2 * x_dot - ((1 - mu) * y) / ((x + mu) ** 2 + y ** 2 + z ** 2) ** (3 / 2) \
                         - (mu * y) / ((x - (1 - mu)) ** 2 + y ** 2 + z ** 2) ** (3 / 2)
                z_ddot = -((1 - mu) * z) / ((x + mu) ** 2 + y ** 2 + z ** 2) ** (3 / 2) \
                         - (mu * z) / ((x - (1 - mu)) ** 2 + y ** 2 + z ** 2) ** (3 / 2)
            elif self.potential == 'Log':
                x_ddot = x + 2 * y_dot - ((1 - mu) * (x + mu)) / ((x + mu) ** 2 + y ** 2 + z ** 2) \
                         - (mu * (x - (1 - mu))) / ((x - (1 - mu)) ** 2 + y ** 2 + z ** 2)
                y_ddot = y - 2 * x_dot - ((1 - mu) * y) / ((x + mu) ** 2 + y ** 2 + z ** 2) \
                         - (mu * y) / ((x - (1 - mu)) ** 2 + y ** 2 + z ** 2)
                z_ddot = -((1 - mu) * z) / ((x + mu) ** 2 + y ** 2 + z ** 2) \
                         - (mu * z) / ((x - (1 - mu)) ** 2 + y ** 2 + z ** 2)

            dstate_dt = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]
            return dstate_dt

        sol = solve_ivp(dynamic_CR3BP, t, state, method='RK45', max_step=self.get_tolerance())
        return sol

    def plot_CR3BP(self, tf, title="", save=False, plot=True):
        orbit = self.calculate_orbit(tf)

        plt.figure()
        plt.plot(-self.bodies[1].pos[0] / self.r_distance, 0, 'o', markersize=10 * (1 - self.mu), color="grey")
        plt.text(- self.bodies[1].pos[0] / self.r_distance, 0, "m1")
        plt.plot(1 - self.bodies[1].pos[0] / self.r_distance, 0, 'o', markersize=10 * self.mu, color="darkgrey")
        plt.text(1 - self.bodies[1].pos[0] / self.r_distance, 0, "m2")
        # plt.text(self.bodies[2].pos[0] / self.r_distance - self.r_distance / 2,
        # self.bodies[2].pos[1] / self.r_distance - self.r_distance / 10,
        # f"{self.bodies[2].pos / self.r_distance}")
        if self.potential == "Kepler":
            Ls = LagrangePoints_Kepler(self.mu)
        elif self.potential == "Log":
            Ls = LagrangePoints_Log(self.mu)
        for e, L in enumerate(Ls):
            plt.plot(L[0], L[1], 'o', color='green')
            plt.text(L[0], L[1], f"$L_{e + 1}$")
        plt.plot(orbit.y[0], orbit.y[1], color='blue')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.title(title)
        if plot:
            plt.show()
        if save:
            name = "Sim_" + title.replace(", $\mu=$", "_mu_").replace(".", "_").replace("$V_N$ ", "Kepler_").replace("Log ", "Log_")
            plt.savefig(name)
            print('saved '+name)

        plt.xlim([min(orbit.y[0])-0.001, max(orbit.y[0])+0.001])
        plt.ylim([min(orbit.y[1])-0.001, max(orbit.y[1])+0.001])

        if plot:
            plt.show()
        if save:
            name = "Sim_focused_" + title.replace(", $\mu=$", "_mu_").replace(".", "_").replace("$V_N$ ", "Kepler_").replace("Log ", "Log_")
            plt.savefig(name)
            print('saved '+name)
