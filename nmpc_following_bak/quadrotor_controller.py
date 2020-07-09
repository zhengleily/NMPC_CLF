import numpy as np
import math
from casadi import *

g = 9.8
m = 0.04 #kg
def quadrotor_controller(s,xl,vl,psi_p=0):
    # pid
    k1 = 1
    k2 = 10

    # state
    x = s[0]
    y = s[1]
    z = s[2]
    vx = s[3]
    vy = s[4]
    vz = s[5]
    phi = s[6]
    theta = s[7]
    psi = s[8]

    # controls
    u = np.zeros(4)

    def saturate(input):
        return np.tanh(input)

    input_delta = k1*(xl-np.array([x, y, z])) + k2*(vl-np.array([vx, vy, vz]))
    attr=saturate(input_delta)*2+np.array([0, 0, g])
    rot=np.array([[np.cos(psi), np.sin(psi), 0],[-np.sin(psi), np.cos(psi), 0],[0, 0, 1]])

    attr = np.dot(rot, np.reshape(attr, [-1, 1]))
    u[0] = m * np.linalg.norm(attr)

    phi_p = math.asin(-attr[1] / np.linalg.norm(attr))
    theta_p = math.atan(attr[0] / attr[2])
    if attr[2] < 0:
        theta_p = theta_p - np.pi
    psi_p = psi_p
    err = np.array([(phi_p - phi), (theta_p - theta), (psi_p - psi)])
    u[1:] = err
    return u

def mpc_attitude_control(s,xl,vl, N, Ts, u_old):
    p_x = SX.sym('p_x', N)
    p_y = SX.sym('p_y', N)
    p_z = SX.sym('p_z', N)
    v_x = SX.sym('v_x', N)
    v_y = SX.sym('v_y', N)
    v_z = SX.sym('v_z', N)
    phi = SX.sym('phi', N)
    theta = SX.sym('theta', N)
    psi = SX.sym('psi', N)

    # controls
    F   = SX.sym('F', N)
    w_x = SX.sym('w_x', N)
    w_y = SX.sym('w_y', N)
    w_z = SX.sym('w_z', N)
    all_vars = vertcat(p_x, p_y, p_z, v_x, v_y, v_z, phi, theta, psi, F, w_x, w_y, w_z)

    ## dynamics function
    p_x_constrain = SX.sym('p_x_constrain', N - 1)
    p_y_constrain = SX.sym('p_y_constrain', N - 1)
    p_z_constrain = SX.sym('p_z_constrain', N - 1)
    v_x_constrain = SX.sym('v_x_constrain', N - 1)
    v_y_constrain = SX.sym('v_y_constrain', N - 1)
    v_z_constrain = SX.sym('v_z_constrain', N - 1)
    phi_constrain = SX.sym('phi_constrain', N - 1)
    theta_constrain = SX.sym('theta_constrain', N - 1)
    psi_constrain = SX.sym('psi_constrain', N - 1)
    for i in range(N - 1):
        p_x_constrain[i] = p_x[i + 1] - (p_x[i] + v_x[i] * Ts)
        p_y_constrain[i] = p_y[i + 1] - (p_y[i] + v_y[i] * Ts)
        p_z_constrain[i] = p_z[i + 1] - (p_z[i] + v_z[i] * Ts)
        v_x_constrain[i] = v_x[i + 1] - (v_x[i] + ((np.cos(psi[i]) * np.sin(theta[i]) * np.cos(phi[i]) + np.sin(psi[i]) * np.sin(phi[i])) * F[i])/m * Ts)
        v_y_constrain[i] = v_y[i + 1] - (v_y[i] + ((np.sin(psi[i]) * np.sin(theta[i]) * np.cos(phi[i]) - np.cos(psi[i]) * np.sin(phi[i])) * F[i])/m * Ts)
        v_z_constrain[i] = v_z[i + 1] - (v_z[i] + ((np.cos(theta[i]) * np.cos(phi[i])) * F[i] - g * m)/m * Ts)
        phi_constrain[i] = phi[i + 1] - (phi[i] + (1 * w_x[i] + np.sin(phi[i]) * np.tan(theta[i]) * w_y[i] + np.cos(phi[i]) * np.tan(theta[i]) * w_z[i]) * Ts)
        theta_constrain[i] = theta[i + 1] - (theta[i] + (0 * w_x[i] + np.cos(phi[i]) * w_y[i] - np.sin(phi[i]) * w_z[i]) * Ts)
        psi_constrain[i] = psi[i + 1] - (psi[i] + (0 * w_x[i] + np.sin(phi[i])/np.cos(theta[i]) * w_y[i] + np.cos(phi[i])/np.cos(theta[i]) * w_z[i]) * Ts)
    all_g_constrain = vertcat(p_x_constrain, p_y_constrain, p_z_constrain, v_x_constrain, v_y_constrain, v_z_constrain, phi_constrain, theta_constrain, psi_constrain)
    ub_constrains_g = np.zeros([9 * (N - 1)])
    lb_constrains_g = np.zeros([9 * (N - 1)])

    # define cost function f
    cost = 0
    for i in range(N):
        # state
        # cost += 100 * (p_x[i] - sDesList[i, 0]) ** 2
        cost += (p_x[i] - xl[0,i]) ** 2
        cost += (p_y[i] - xl[1,i]) ** 2
        cost += (p_z[i] - xl[2,i]) ** 2
        # cost += 0.05 * (v_x[i] - vl[0]) ** 2
        # cost += 0.05 * (v_y[i] - vl[1]) ** 2
        # cost += 0.05 * (v_z[i] - vl[2]) ** 2
        # if i < N-1:
        #     # control
        #     cost += 1 * (F[i+1] - F[i]) ** 2
        #     cost += 1 * (w_x[i+1] - w_x[i]) ** 2
        #     cost += 1 * (w_y[i+1] - w_y[i]) ** 2
        #     cost += 1 * (w_z[i+1] - w_z[i]) ** 2

    uMax = 0.55 # N
    uMin = 0.
    pMax = 10.
    qMax = 10.
    rMax = 10.
    # vars upper bound
    ub_constrains_p_x = np.array([np.inf] * N)
    ub_constrains_p_y = np.array([np.inf] * N)
    ub_constrains_p_z = np.array([np.inf] * N)
    ub_constrains_v_x = np.array([np.inf] * N)
    ub_constrains_v_y = np.array([np.inf] * N)
    ub_constrains_v_z = np.array([np.inf] * N)
    ub_constrains_phi = np.array([np.inf] * N)
    ub_constrains_theta = np.array([np.inf] * N)
    ub_constrains_psi = np.array([np.inf] * N)
    ub_constrains_F   = np.array([uMax] * N)
    ub_constrains_w_x = np.array([pMax] * N)
    ub_constrains_w_y = np.array([qMax] * N)
    ub_constrains_w_z = np.array([rMax] * N)
    ub_constrains_vars = np.hstack([ub_constrains_p_x, ub_constrains_p_y, ub_constrains_p_z,\
                    ub_constrains_v_x, ub_constrains_v_y, ub_constrains_v_z, \
                    ub_constrains_phi, ub_constrains_theta, ub_constrains_psi, \
                    ub_constrains_F, ub_constrains_w_x, ub_constrains_w_y, ub_constrains_w_z])
    # vars lower bound
    lb_constrains_p_x = np.array([-np.inf] * N)
    lb_constrains_p_y = np.array([-np.inf] * N)
    lb_constrains_p_z = np.array([-np.inf] * N)
    lb_constrains_v_x = np.array([-np.inf] * N)
    lb_constrains_v_y = np.array([-np.inf] * N)
    lb_constrains_v_z = np.array([-np.inf] * N)
    lb_constrains_phi = np.array([-np.inf] * N)
    lb_constrains_theta = np.array([-np.inf] * N)
    lb_constrains_psi = np.array([-np.inf] * N)
    lb_constrains_F   = np.array([uMin] * N)
    lb_constrains_w_x = np.array([-pMax] * N)
    lb_constrains_w_y = np.array([-qMax] * N)
    lb_constrains_w_z = np.array([-rMax] * N)
    lb_constrains_vars = vertcat(lb_constrains_p_x, lb_constrains_p_y, lb_constrains_p_z,\
                    lb_constrains_v_x, lb_constrains_v_y, lb_constrains_v_z, \
                    lb_constrains_phi, lb_constrains_theta, lb_constrains_psi, \
                    lb_constrains_F, lb_constrains_w_x, lb_constrains_w_y, lb_constrains_w_z)


    # state
    x = s[0]
    y = s[1]
    z = s[2]
    vx = s[3]
    vy = s[4]
    vz = s[5]
    phi = s[6]
    theta = s[7]
    psi = s[8]
    p_x_init = [x] * N
    p_y_init = [y] * N
    p_z_init = [z] * N
    v_x_init = [vx] * N
    v_y_init = [vy] * N
    v_z_init = [vz] * N
    phi_init = [phi] * N
    theta_init = [theta] * N
    psi_init = [psi] * N
    # u: [T, omega]
    T_init = [u_old[0]] * N
    w_x_init = [u_old[1]] * N
    w_y_init = [u_old[2]] * N
    w_z_init = [u_old[3]] * N
    # init_state
    initial_value = p_x_init + p_y_init + p_z_init + v_x_init + v_y_init + v_z_init +\
                    phi_init + theta_init + psi_init + T_init + w_x_init + w_y_init + w_z_init

    # constrain the first variable to match with the init

    ub_constrains_vars[0 * N] = p_x_init[0]
    ub_constrains_vars[1 * N] = p_y_init[0]
    ub_constrains_vars[2 * N] = p_z_init[0]
    ub_constrains_vars[3 * N] = v_x_init[0]
    ub_constrains_vars[4 * N] = v_y_init[0]
    ub_constrains_vars[5 * N] = v_z_init[0]
    ub_constrains_vars[6 * N] = phi_init[0]
    ub_constrains_vars[7 * N] = theta_init[0]
    ub_constrains_vars[8 * N] = psi_init[0]

    lb_constrains_vars[0 * N] = p_x_init[0]
    lb_constrains_vars[1 * N] = p_y_init[0]
    lb_constrains_vars[2 * N] = p_z_init[0]
    lb_constrains_vars[3 * N] = v_x_init[0]
    lb_constrains_vars[4 * N] = v_y_init[0]
    lb_constrains_vars[5 * N] = v_z_init[0]
    lb_constrains_vars[6 * N] = phi_init[0]
    lb_constrains_vars[7 * N] = theta_init[0]
    lb_constrains_vars[8 * N] = psi_init[0]

    nlp =  {'x': all_vars,
            'f': cost,
            'g': all_g_constrain}

    S = nlpsol('S', 'ipopt', nlp, {"print_time": False,
                                    "ipopt":   {"print_level": 0,
                                                "print_timing_statistics": "yes",
                                                "max_cpu_time": 1}})

    result = S(x0=initial_value,        lbg=lb_constrains_g,   ubg=ub_constrains_g,\
              lbx=lb_constrains_vars,   ubx=ub_constrains_vars)

    return np.array([result['x'][9 * N], result['x'][10 * N], result['x'][11 * N ], result['x'][12 * N]]), \
           np.array([result['x'][9 * N + 1], result['x'][10 * N + 1], result['x'][11 * N + 1], result['x'][12 * N + 1]])
    # return np.array([self.env.quad.g * self.env.quad.m, 0., 0., 0.])


