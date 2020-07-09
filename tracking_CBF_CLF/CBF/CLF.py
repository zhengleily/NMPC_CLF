

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from CBF.GP import GP
import math
import irispy


# Build barrier function model
def build_barrier(self):
    N = self.action_space
    self.P_angle = matrix(np.diag([1., 1., 1., 1., 1e20, 1e21]), tc='d')
    self.P_pos = matrix(np.diag([1., 1., 1., 1., 1e20]), tc='d')
    self.q_angle = matrix(np.zeros(N + 2))
    self.q_pos = matrix(np.zeros(N + 1))


def cal_angle(self, state, state_dsr, gp_prediction):
    pos_vec, vel_vec = self.env.direction(state, state_dsr, gp_prediction)
    pos_vec = pos_vec / np.linalg.norm(pos_vec)
    vel_vec = vel_vec / np.linalg.norm(vel_vec)
    psi = [0]
    phi = np.arcsin(-pos_vec[1])
    theta = np.arcsin((pos_vec[0] / np.cos(phi)))
    pos_angle = np.concatenate([phi, theta, psi])
    phi = np.arcsin(-vel_vec[1])
    theta = np.arcsin((vel_vec[0] / np.cos(phi)))
    vel_angle = np.concatenate([phi, theta, psi])
    return pos_angle, vel_angle


# Get compensatory action based on satisfaction of barrier function
def control_barrier(self, obs, u_rl, f, g, x, t):
    step_time = t
    dt = self.env.dt
    ''' Recorded curve '''
    # curve = self.env.curve
    # cur_pt = curve[t,:]
    # next_pt = curve[t+1,:]
    # third_pt = curve[t+2,:]
    ''' Parametric curve '''
    cur_pt = np.concatenate([self.env.curve_pos(step_time * dt), self.env.curve_vel(step_time * dt)])
    next_pt = np.concatenate([self.env.curve_pos((step_time + 1) * dt), self.env.curve_vel((step_time + 1) * dt)])
    third_pt = np.concatenate([self.env.curve_pos((step_time + 2) * dt), self.env.curve_vel((step_time + 2) * dt)])

    # Set up Quadratic Program to satisfy the Control Barrier Function
    kd = 2.0
    # calculate energy function V_t
    v_t_pos = np.abs(cur_pt - obs[:6])
    v_t_angle = np.abs(self.env.cur_angle - obs[6:9])
    N = self.observation_space
    M = self.action_space
    '''The main hyperparameter'''
    # gamma 越小（甚至为0）,能量衰减越慢, 震荡越小, 设置过小会导致跟踪收敛慢
    gamma_pos_clf = 0.1
    gamma_angle_clf = 0.1
    # gamma_pos_cbf 越小, 越不允许往更不安全的地方探索, 因此应根据实际场景调整设置
    gamma_pos_cbf = 100
    # gamma_angle_cbf 设置过小则会对角度调整过于敏感
    gamma_angle_cbf = 0.01

    up_b = self.action_bound_up.reshape([M, 1])  ##
    low_b = self.action_bound_low.reshape([M, 1])  ##
    f = np.reshape(f, [N, 1])
    g = np.reshape(g, [N, M])
    # std = np.reshape(std, [N, 1])
    std = np.zeros([N, 1])
    u_rl = np.reshape(u_rl, [M, 1])  #
    x = np.reshape(x, [N, 1])

    '''QP 1 : use CBF to solve thrust'''
    '''h1(x) = (x-x_b)^2 + (y-y_b)^2 + (z-z_b)^2 - belta * b^2'''
    '''Multiple static obstacles'''
    if (step_time) % 2000 == 0:
        self.obstacles = []
        center1 = np.array(
            [[-2 * np.sin(0.5 * 620 * 0.02), 5 + 2 * np.cos(0.5 * 620 * 0.02), .4 * 620 * 0.02]]).reshape([3, 1])
        center2 = np.array(
            [[-2 * np.sin(0.5 * 520 * 0.02) - 0.3, 4.15 + 2 * np.cos(0.5 * 600 * 0.02), .4 * 580 * 0.02]]).reshape(
            [3, 1])
        center3 = np.array(
            [[-2 * np.sin(0.5 * 520 * 0.02) - 0.35, 4.6 + 2 * np.cos(0.5 * 600 * 0.02), .4 * 590 * 0.02]]).reshape(
            [3, 1])
        center4 = np.array(
            [[-2 * np.sin(0.5 * 600 * 0.02), 5 + 2 * np.cos(0.5 * 610 * 0.02), .4 * 520 * 0.02]]).reshape([3, 1])
        scale = 1
        pts = np.min([np.random.random([3, 8]) * 0.8, 0.5 * np.ones([3, 8])], axis=0)
        pts = np.max([pts, 0.2 * np.ones([3, 8])], axis=0)
        pts = pts - np.mean(pts, axis=1)[:, np.newaxis]
        pts1 = scale * pts + center1
        pts2 = scale * pts + center2
        pts3 = scale * pts + center3
        pts4 = scale * pts + center4
        self.obstacles.append(pts1)
        self.obstacles.append(pts2)
        self.obstacles.append(pts3)
        self.obstacles.append(pts4)
    '''Multiple random generated static obstacles'''
    # if (step_time) % 20 == 0:
    #     self.obstacles = []
    #     self.center = []
    #     for i in range(15):
    #         center = np.random.random(3,) * np.array([2,2,2.5]) + np.array([-2*np.sin(0.5*step_time*0.02)-0.5, 5 + 2*np.cos(0.5*step_time*0.02)-0.5, .4*step_time*0.02-1.5])
    #         scale = 1
    #         pts = np.min([np.random.random([3, 8])*0.5, 0.5*np.ones([3, 8])], axis=0)
    #         pts = np.max([pts, 0.1*np.ones([3, 8])], axis=0)
    #         a = np.mean(pts, axis=1)[:, np.newaxis]
    #         pts = pts - np.mean(pts, axis=1)[:, np.newaxis]
    #         pts = scale * pts + center[:, np.newaxis]
    #         self.obstacles.append(pts)
    #         self.center.append(center)
    # self.env.barrier = np.array(self.center).reshape([-1,3])
    if (step_time) % 1 == 0:
        bounds = irispy.Polyhedron.from_bounds(obs[:3] - 1.5 * self.env.quad.l * np.ones(3, ),
                                               obs[:3] + 1.5 * self.env.quad.l * np.ones(3, ))
        self.env.obstacles = self.obstacles
        start = obs[:3]
        region, debug = irispy.inflate_region(self.obstacles, start, bounds=bounds, return_debug_data=True)
        iter_result = debug.iterRegions()
        ellipsoid = list(iter_result)[-1][1]
        mapping_matrix = ellipsoid.getC() * 0.9
        center_pt = ellipsoid.getD().reshape([-1, 1])
        inv_mapping_matrix = np.linalg.inv(mapping_matrix)
        A_mat = np.dot(inv_mapping_matrix.T, inv_mapping_matrix)
        b_vec = -2 * A_mat.dot(center_pt)
        c = np.dot(center_pt.T, A_mat).dot(center_pt)
        if (1 - x[:3, :].T.dot(A_mat).dot(x[:3, :]) - b_vec.T.dot(x[:3, :]) - c) < 0:
            gamma_pos_cbf = 100
            print('out of safe region')
        else:
            gamma_pos_cbf = (1 - x[:3, :].T.dot(A_mat).dot(x[:3, :]) - b_vec.T.dot(x[:3, :]) - c)
        self.safe_region = [A_mat, b_vec, c]
        self.ellipsoid = [mapping_matrix, center_pt]

    [A_mat, b_vec, c] = self.safe_region
    print(1 - x[:3, :].T.dot(A_mat).dot(x[:3, :]) - b_vec.T.dot(x[:3, :]) - c)

    G_pos = np.concatenate([np.concatenate([
        (2 * A_mat.dot(x[:3, :]) + b_vec) * g[:3, :],
        np.eye(M), -np.eye(M)], axis=0),
        # np.concatenate([np.zeros([2 * M, 1])], axis=0),
        # np.concatenate([np.zeros([2 * M, 1])], axis=0),
        np.concatenate([-np.ones([3, 1]), np.zeros([2 * M, 1])], axis=0)], axis=1)

    h_pos = np.concatenate([
        # np.ones([1, 1]) * (gamma_pos_cbf * dt * (1 - x[:3,:].T.dot(A_mat).dot(x[:3,:]) - b_vec.T.dot(x[:3,:]) - c) - (2 * A_mat.dot(x[:3,:]) + b_vec) * (f[:3,:] - x[:3,:] - abs(kd*std[:3,:]))),
        np.ones([1, 1]) * (-np.dot((2 * A_mat.dot(x[:3, :]) + b_vec) * g[:3, :], u_rl) + gamma_pos_cbf * dt * (
                    1 - x[:3, :].T.dot(A_mat).dot(x[:3, :]) - b_vec.T.dot(x[:3, :]) - c) - (
                                       2 * A_mat.dot(x[:3, :]) + b_vec) * (f[:3, :] - x[:3, :]) - abs(
            (2 * A_mat.dot(x[:3, :]) + b_vec) * kd * std[:3, :])),
        up_b,
        -low_b], axis=0)
    G_pos = matrix(G_pos, tc='d')
    h_pos = matrix(h_pos, tc='d')
    # Solve QP
    solvers.options['show_progress'] = False
    sol = solvers.qp(P=self.P_pos, q=self.q_pos, G=G_pos, h=h_pos)
    thrust = sol['x'][0] + u_rl[0]
    # predict new pos and vel to calculate new angle
    predict_xyz = f + np.dot(g, thrust * np.ones([4, 1]))
    gp_prediction = GP.get_GP_prediction(self, predict_xyz[:6])
    # gp_prediction = np.zeros(6)
    pos_angle, vel_angle = cal_angle(self, predict_xyz[:6].squeeze(), third_pt, gp_prediction)
    weight = 0.7
    [phi_d, theta_d, psi_d] = weight * pos_angle + (1 - weight) * vel_angle
    self.env.cur_angle = np.array([phi_d, theta_d, psi_d])

    '''QP 2 : use CLF to approximate the desired angle as well as CBF to escape from the obstacle'''
    '''v2(phi) = |phi-phi_d| ,v2(theta) = |theta-theta_d|, v2(psi) = |psi-psi_d|'''
    '''h2(x) = (x-x_b)^2 + (y-y_b)^2 + (z-z_b)^2 - belta * b^2 + sigma(r*q)'''
    # d = 1
    # if d <= 0.5:
    r = x[:3, :] - center_pt
    q = np.array([math.cos(x[6]) * math.sin(x[7]), -math.sin(x[6]), math.cos(x[6]) * math.cos(x[7])]).reshape(([3, 1]))
    r_q = np.squeeze(np.dot(r.T, q))
    r_dot = (predict_xyz[:3, :] - x[:3, :]) / dt
    # a1, a2, a3 = 2 * (belta-1) * safety_bound ** 2 / np.pi, 1, 0
    # sigma = lambda s: a1 * math.atan(a2 * s + a3)
    # sigma_dot = lambda s: a1 * a2 / (1 + (a2 * s + a3) ** 2)
    # sigma_r_q = sigma(r_q)[0]
    # sigma_dot_r_q = sigma_dot(r_q)[0]
    q_dot_mat = np.array([
        [-math.sin(x[6]) * math.sin(x[7]), math.cos(x[6]) * math.cos(x[7]), 0],
        [-math.cos(x[6]), 0, 0],
        [-math.sin(x[6]) * math.cos(x[7]), -math.cos(x[6]) * math.sin(x[7]), 0]
    ])
    r_q_dot_mat = np.dot(r.T, q_dot_mat)

    G_angle = np.concatenate([np.concatenate([
        g[6, :].reshape([1, -1]),
        -g[6, :].reshape([1, -1]),
        g[7, :].reshape([1, -1]),
        -g[7, :].reshape([1, -1]),
        g[8, :].reshape([1, -1]),
        -g[8, :].reshape([1, -1]),
        -np.dot(r_q_dot_mat, g[6:, :]).reshape([1, -1]),
        np.eye(M), -np.eye(M)], axis=0),
        np.concatenate([-np.ones([6, 1]), np.zeros([1, 1]), np.zeros([2 * M, 1])], axis=0),
        np.concatenate([np.zeros([6, 1]), -np.ones([1, 1]), np.zeros([2 * M, 1])], axis=0),
    ], axis=1)
    '''use MPC framework'''
    h_angle = np.concatenate([
        np.ones([1, 1]) * ((1 - gamma_angle_clf) * v_t_angle[0] - f[6] + phi_d - np.dot(g[6, :].reshape([1, M]), u_rl)),
        np.ones([1, 1]) * ((1 - gamma_angle_clf) * v_t_angle[0] + f[6] - phi_d + np.dot(g[6, :].reshape([1, M]), u_rl)),
        np.ones([1, 1]) * (
                    (1 - gamma_angle_clf) * v_t_angle[1] - f[7] + theta_d - np.dot(g[7, :].reshape([1, M]), u_rl)),
        np.ones([1, 1]) * (
                    (1 - gamma_angle_clf) * v_t_angle[1] + f[7] - theta_d + np.dot(g[7, :].reshape([1, M]), u_rl)),
        np.ones([1, 1]) * ((1 - gamma_angle_clf) * v_t_angle[2] - f[8] + psi_d - np.dot(g[8, :].reshape([1, M]), u_rl)),
        np.ones([1, 1]) * ((1 - gamma_angle_clf) * v_t_angle[2] + f[8] - psi_d + np.dot(g[8, :].reshape([1, M]), u_rl)),
        np.ones([1, 1]) * ((gamma_angle_cbf * r_q * dt + np.dot(r_dot.T, q)) * dt
                           + np.dot(r_q_dot_mat, (f[6:, :] + np.dot(g[6:, :], u_rl) - x[6:, :]))),
        -u_rl + up_b,
        u_rl - low_b], axis=0)
    '''Don't use MPC framework'''
    # h_angle = np.concatenate([
    #     np.ones([1, 1]) * ((1 - gamma_angle_clf) * v_t_angle[0] - f[6] + phi_d),
    #     np.ones([1, 1]) * ((1 - gamma_angle_clf) * v_t_angle[0] + f[6] - phi_d),
    #     np.ones([1, 1]) * (
    #                 (1 - gamma_angle_clf) * v_t_angle[1] - f[7] + theta_d),
    #     np.ones([1, 1]) * (
    #                 (1 - gamma_angle_clf) * v_t_angle[1] + f[7] - theta_d),
    #     np.ones([1, 1]) * ((1 - gamma_angle_clf) * v_t_angle[2] - f[8] + psi_d),
    #     np.ones([1, 1]) * ((1 - gamma_angle_clf) * v_t_angle[2] + f[8] - psi_d),
    #     np.ones([1, 1]) * ((gamma_angle_cbf * r_q * dt + np.dot(r_dot.T, q)) * dt
    #                        + np.dot(r_q_dot_mat, (f[6:, :] - x[6:, :]))),
    #     -u_rl + up_b,
    #     u_rl - low_b], axis=0)

    h_angle = np.squeeze(h_angle).astype(np.double)
    # Convert numpy arrays to cvx matrices to set up QP
    G_angle = matrix(G_angle, tc='d')
    h_angle = matrix(h_angle, tc='d')
    # Solve QP
    solvers.options['show_progress'] = False
    sol = solvers.qp(self.P_angle, self.q_angle, G=G_angle, h=h_angle)
    u_bar = np.squeeze(sol['x'])
    # Torque bound violation
    # u_bar[:-3] = np.min([np.squeeze(up_b - u_rl), u_bar[:-3]], axis=0)
    # u_bar[:-3] = np.max([np.squeeze(low_b - u_rl), u_bar[:-3]], axis=0)
    u_bar[0] = thrust - u_rl[0]
    # np.set_printoptions(suppress=True)
    # print('------------------')
    # print('cur_angle',obs[6:9])
    # print('predict_next_angle',(f + np.dot(g, u_rl+u_bar[:4].reshape([4,1]))).squeeze()[6:9].astype(np.float))
    # print('next_angle',phi_d,theta_d,psi_d)
    # print('cur_xyz',obs[:6].astype(np.float))
    # print('predict_next_xyz',(f + np.dot(g, u_rl+u_bar[:4].reshape([4,1]))).squeeze()[:6].astype(np.float))
    # print('next_xyz',next_pt[:6].astype(np.float))
    # print('------------------')

    return np.expand_dims(np.array(u_bar[:4]), 0), np.sum(v_t_pos), np.sum(v_t_angle)
