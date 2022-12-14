from Bodies import BodySystem, Body
from scipy.integrate import solve_ivp
from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
from aux_tools import pol2car


def theoretical_orbit(h, mu, e, theta):
    r = (h*h/mu)/(1+e*np.cos(theta))
    return r


class TwoBodySystem(BodySystem):
    def __init__(self, G=None, body1=None, body2=None, potential=None):
        super().__init__(G)
        self.potential = potential
        self.bodies = [body1, body2]
        self.r_cg = self._calculate_cg()
        self.r = self.bodies[1].pos - self.bodies[0].pos
        self.r_distance = np.linalg.norm(self.r)
        self.cg_percentaje = np.linalg.norm(self._calculate_cg() - body1.pos) / self.r_distance
        self.v = self.bodies[1].vel - self.bodies[0].vel
        self.vr_module = np.linalg.norm(np.dot(self.r, self.v) / self.r_distance)

        self.reduced_mass = np.prod([x.mass for x in self.bodies]) / np.sum(
            x.mass for x in self.bodies)  # reduced mass from wikipedia
        self.mu = self.G * np.sum(x.mass for x in self.bodies)
        self.spec_angular_momemtum = self._specific_angular_momentum(self.r, self.v)
        self.spec_angular_momemtum_module = np.linalg.norm(self.spec_angular_momemtum)
        self.Vef = self.V_ef(self.r_distance)
        self.spec_mec_energy = self._specific_mechanical_energy(self.vr_module, self.Vef)
        self.eccentricity = self._orbit_eccentricity()
        self._tolerance = 0.0001

        self.rel_polar_orbit_2 = self.calculate_orbit(0, 2 * np.pi)
        self.rel_polar_cg = self.cg_percentaje * self.rel_polar_orbit_2[0], self.rel_polar_orbit_2[1]

        self.rel_orbit_2 = pol2car(self.rel_polar_orbit_2[0], self.rel_polar_orbit_2[1])
        self.rel_cg = pol2car(self.rel_polar_cg[0], self.rel_polar_cg[1])

        # respect to CG (_recalculate_orbit_cg)
        self.orbit_1, self.orbit_2 = self._recalculate_orbit_cg()

    def set_tolerance(self, tol):
        self._tolerance = tol

    def get_tolerance(self):
        return self._tolerance

    def calculate_orbit(self, theta0, thetafin):
        # take initial conditions
        r0 = self.r_distance
        drdt0 = self.vr_module  # radial velocity projection
        h = self.spec_angular_momemtum_module
        u0 = [1 / r0, drdt0 / (-h)]

        # theta_span = np.linspace(theta0, thetafin, int(1/self.get_tolerance()))
        theta_span = (theta0, thetafin)

        # define dynamic equations
        def dynamic(theta, u):
            y = np.zeros((2, 1)).reshape((2,))
            y[0] = u[1]
            dduddtheta = 0
            if self.potential == "Newton":
                dduddtheta = -u[0] + self.mu / (h * h)
            elif self.potential == "Log":
                dduddtheta = -u[0] + self.mu / (h * h * u[0])
            else:
                print("Error selecting potential: Newton or Log")
            y[1] = dduddtheta
            return y

        sol = solve_ivp(dynamic, theta_span, u0, method='RK45', max_step=self.get_tolerance())
        sol_r = 1 / sol.y[0]
        sol_theta = sol.t
        return sol_r, sol_theta

    def update_orbit(self, theta0, thetafin):
        self.rel_polar_orbit_2 = self.calculate_orbit(theta0, thetafin)
        self.rel_polar_cg = self.cg_percentaje * self.rel_polar_orbit_2[0], self.rel_polar_orbit_2[1]

        self.rel_orbit_2 = pol2car(self.rel_polar_orbit_2[0], self.rel_polar_orbit_2[1])
        self.rel_cg = pol2car(self.rel_polar_cg[0], self.rel_polar_cg[1])

        # respect to CG (_recalculate_orbit_cg)
        self.orbit_1, self.orbit_2 = self._recalculate_orbit_cg()

    def _calculate_cg(self):
        cg = 0
        total_mass = 0
        for x in self.bodies:
            cg += x.mass * x.pos
            total_mass += x.mass
        return cg / total_mass

    def _recalculate_orbit_cg(self):  # origin from cg
        orbit_2 = self.rel_orbit_2 - self.rel_cg
        orbit_1 = -self.rel_cg
        return np.array([orbit_1, orbit_2])

    def _orbit_eccentricity(self):
        return np.sqrt(1 + 2 * self.spec_angular_momemtum_module ** 2 * self.spec_mec_energy / (self.mu ** 2))
        # return np.linalg.norm(np.cross(self.v, self.spec_angular_momemtum)/self.mu - self.r/np.linalg.norm(self.r))

    def _specific_mechanical_energy(self, v, pot):
        return v ** 2 / 2 + pot

    def V_Newton(self, r_mod):
        return - self.mu / r_mod

    def V_log(self, r_mod):
        return self.mu * np.log(r_mod)

    def V_ef(self, r_mod):
        h = self.spec_angular_momemtum_module
        if self.potential == 'Newton':
            return 1 / 2 * (h / r_mod) ** 2 + self.V_Newton(r_mod)
        elif self.potential == 'Log':
            return 1 / 2 * (h / r_mod) ** 2 + self.V_log(r_mod)  # * 1/2*np.linalg.norm(self.v)**2
        else:
            print("Error choosing potential name: Newton or Log")

    def _specific_angular_momentum(self, r, v):
        return np.cross(r, v)

    def plot_Enery_potential(self):
        r0 = self.r_distance / 1e6
        rfin = self.r_distance * 2
        r = np.linspace(r0, rfin, int(1e7))
        E = self.spec_mec_energy
        Vef = self.V_ef(r)
        E_line = self.spec_mec_energy * np.ones(r.size)

        def f1(x):
            sol = self.V_ef(x) - E
            return sol

        def f2(x):
            sol = E - self.V_ef(x)
            return sol

        if self.potential == "Newton":
            f = f1
            minimum = optimize.minimize(f, x0=r0, method="Nelder-Mead")
            m = minimum.x[0]
            r_point1 = (optimize.Newton(f, m - m / 2, maxiter=100000), 0)
            r_point2 = (optimize.Newton(f, m + r0 / 2, maxiter=100000), 0)
        elif self.potential == "Log":
            f = f2
            m = 12600
            r_point1 = (optimize.Newton(f, 5, maxiter=100000), 0)
            r_point2 = (optimize.Newton(f, 10, maxiter=100000), 0)

        preplot = False

        if preplot:
            auxfig = plt.figure()
            # plt.plot(r, f(r))
            plt.plot(r, self.V_ef(r))
            plt.plot(r, E_line)
            plt.axis([-0.5, 20, -5, 10])
            plt.show()

        fig_energy = plt.figure(10)
        plt.axhline(0, color='grey')
        plt.text(r_point1[0], 0, str(r_point1[0]), rotation='vertical')
        plt.axvline(r_point1[0], color='r')
        plt.text(r_point2[0], 0, str(r_point2[0]), rotation='vertical')
        plt.axvline(r_point2[0], color='r')
        plt.plot(r, E_line, label="E")
        plt.plot(r, Vef, label='Vef')
        plt.plot(r, f(r), '--', label='Vef-E')
        if self.potential == "Log":
            plt.axis([-100, r_point2[0] / 1e5, min(f(r)), E])
        plt.xlabel('r')
        plt.legend()
        plt.show()
        fig_energy.savefig(f'Energy_Vef_{self.potential}')
