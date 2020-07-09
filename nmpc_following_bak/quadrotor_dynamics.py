import numpy as np

m = 0.04 # kg
g = 9.8

class Quad():
    def __init__(self):
        self.g = 9.81
        self.m = 0.04
    def quadrotor_dynamics(self, s, u, dt):
        '''
        Dynamics for quad
        '''
        x = s[0]
        y = s[1]
        z = s[2]
        vx = s[3]
        vy = s[4]
        vz = s[5]
        phi = s[6]
        theta = s[7]
        psi = s[8]

        w_x = u[1]
        w_y = u[2]
        w_z = u[3]
        wind = self.wind(x, y, z)
        Rotation = np.array([
            np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi),
            np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi),
            np.cos(theta) * np.cos(phi)
        ])

        accel = np.array([0, 0, -self.g]) + (Rotation * u[0] + wind) /  self.m
        vel = np.array([vx, vy, vz]) + accel * dt
        position = np.array([x, y, z]) + vel * dt

        # ax = u[0] * (np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)) / m
        # vx = vx + dt * ax
        # x = x + dt * vx
        # ay = u[0] * (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)) / m
        # vy = vy + dt * ay
        # y = y + dt * vy
        # az = u[0] * (np.cos(theta) * np.cos(phi)) / m - g
        # vz = vz + dt * az
        # z = z + dt * vz

        vphi = 1 * w_x + np.sin(phi) * np.tan(theta) * w_y + np.cos(phi) * np.tan(theta) * w_z
        vtheta = 0 * w_x + np.cos(phi) * w_y - np.sin(phi) * w_z
        vpsi = 0 * w_x + np.sin(phi) / np.cos(theta) * w_y + np.cos(phi) / np.cos(theta) * w_z
        phi = phi + vphi * dt
        theta = theta + vtheta * dt
        psi = psi + vpsi * dt
        # s = np.array([x, y, z, vx, vy,  vz, phi, theta, psi])
        s = np.array([position[0], position[1], position[2], vel[0], vel[1],  vel[2], phi, theta, psi])

        return s

    def wind(self, x, y, z):
        v = [
            0*0.15,
            0*0.15,
            0*0.05,
        ]
        return np.array(v)
