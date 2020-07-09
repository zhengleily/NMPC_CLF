#!/usr/bin/env python3
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
    self.P_pos = matrix(np.diag([1., 1., 1., 1., 1e20,  1e19, 1e30]), tc='d')
    self.q_angle = matrix(np.zeros(N + 2))
    self.q_pos = matrix(np.zeros(N + 3))
    self.x_max = 0.05
    self.y_max = 0.05
    self.z_max = 0.05


def cal_angle(self, state, state_dsr,gp_prediction):
    pos_vec,vel_vec = self.env.direction(state, state_dsr,gp_prediction)
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
def control_barrier(self, obs, u_rl, f, g, std, x_nominal, t):
    step_time = t
    dt = self.env.dt
    ''' Recorded curve '''
    # curve = self.env.curve
    # cur_pt = curve[t,:]
    # next_pt = curve[t+1,:]
    # third_pt = curve[t+2,:]
    ''' Parametric curve '''
    print('time_step:',t)
    if t == 144:
        print('quit')
    cur_pt = np.concatenate([self.env.curve_pos(step_time*dt), self.env.curve_vel(step_time*dt)])
    # next_pt = x_nominal
    next_pt = np.concatenate([self.env.curve_pos((step_time + 1)*dt), self.env.curve_vel((step_time + 1)*dt)])
    third_pt = np.concatenate([self.env.curve_pos((step_time + 2)*dt), self.env.curve_vel((step_time + 2)*dt)])

    # Set up Quadratic Program to satisfy the Control Barrier Function
    kd = 2.0
    # calculate energy function V_t
    v_t_pos = np.abs(cur_pt - obs[:6])
    # v_t_pos = np.abs(next_pt - obs[:6])
    v_t_angle = np.abs(self.env.cur_angle - obs[6:9])
    N = self.observation_space
    M = self.action_space
    '''The main hyperparameter'''
    # gamma 越小（甚至为0）,能量衰减越慢, 震荡越小, 设置过小会导致跟踪收敛慢
    gamma_pos_clf = 0.9
    gamma_angle_clf = 0.2
    # gamma_pos_cbf 越小, 越不允许往更不安全的地方探索, 因此应根据实际场景调整设置
    gamma_pos_cbf = 0.4
    # gamma_angle_cbf 设置过小则会对角度调整过于敏感
    gamma_angle_cbf = 0.01

    up_b = self.action_bound_up.reshape([M, 1])##
    low_b = self.action_bound_low.reshape([M, 1])##
    f = np.reshape(f, [N, 1])
    g = np.reshape(g, [N, M])
    std = np.reshape(std, [N, 1])
    # std = np.zeros([N, 1])
    u_rl = np.reshape(u_rl, [M, 1])#


    '''QP 1 : use CLF & CBF to solve thrust ;Using CBF to constrain the state in a tube'''
    '''v1(x) = |x-x_g| ,v1(y) = |y-y_g|, v1(z) = |z-z_g|'''
    '''h1(x) = 1- (x-x_n)^2/x_max - (y-y_n)^2/y_max - (z-z_n)^2/z_max'''
    h_x = 1 - (obs[0] - cur_pt[0])**2/self.x_max**2 - (obs[1] - cur_pt[1])**2/self.y_max **2 - (obs[2] - cur_pt[2])**2/self.z_max**2
    G_pos = np.concatenate([np.concatenate([
    g[0, :].reshape([1, -1]),
    -g[0, :].reshape([1, -1]),
    g[1, :].reshape([1, -1]),
    -g[1, :].reshape([1, -1]),
    g[2, :].reshape([1, -1]),
    -g[2, :].reshape([1, -1]),
    g[3, :].reshape([1, -1]),
    -g[3, :].reshape([1, -1]),
    g[4, :].reshape([1, -1]),
    -g[4, :].reshape([1, -1]),
    g[5, :].reshape([1, -1]),
    -g[5, :].reshape([1, -1]),
    2 * ((obs[0] - next_pt[0]) * g[3, :] / self.x_max ** 2 - (obs[1] - next_pt[1]) * g[4, :] / self.y_max ** 2
         - (obs[2] - next_pt[2]) * g[5, :] / self.z_max ** 2).reshape([1, -1]),
    np.eye(M), -np.eye(M)], axis=0),
    np.concatenate([-np.ones([6, 1]), np.zeros([6, 1]), np.zeros([1, 1]), np.zeros([2 * M, 1])], axis=0),
    np.concatenate([np.zeros([6, 1]), -np.ones([6, 1]), np.zeros([1, 1]), np.zeros([2 * M, 1])], axis=0),
    np.concatenate([np.zeros([6, 1]), np.zeros([6, 1]), -np.ones([1, 1]), np.zeros([2 * M, 1])], axis=0),
    ], axis=1)

    h_pos = np.concatenate([
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[0] - f[0] + next_pt[0]) - abs(kd * std[0]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[0] + f[0] - next_pt[0]) - abs(kd * std[0]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[1] - f[1] + next_pt[1]) - abs(kd * std[1]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[1] + f[1] - next_pt[1]) - abs(kd * std[1]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[2] - f[2] + next_pt[2]) - abs(kd * std[2]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[2] + f[2] - next_pt[2]) - abs(kd * std[2]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[3] - f[3] + next_pt[3]) - abs(kd * std[3]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[3] + f[3] - next_pt[3]) - abs(kd * std[3]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[4] - f[4] + next_pt[4]) - abs(kd * std[4]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[4] + f[4] - next_pt[4]) - abs(kd * std[4]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[5] - f[5] + next_pt[5]) - abs(kd * std[5]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[5] + f[5] - next_pt[5]) - abs(kd * std[5]),
        np.ones([1, 1]) * ((obs[0] - next_pt[0]) * f[3, :] / self.x_max ** 2 - (obs[1] - next_pt[1]) * f[4,:] / self.y_max ** 2
                           - (obs[2] - next_pt[2]) * f[5, :] / self.z_max ** 2 + gamma_pos_cbf * h_x),
        up_b,
        -low_b], axis=0)
    G_pos = matrix(G_pos, tc='d')
    h_pos = matrix(h_pos, tc='d')
    # Solve QP
    solvers.options['show_progress'] = False
    sol = solvers.qp(P=self.P_pos, q=self.q_pos, G=G_pos, h=h_pos)
    thrust = sol['x'][0]
    # predict new pos and vel to calculate new angle
    predict_xyz = f + np.dot(g, thrust * np.ones([4, 1]))
    actual_pos = np.squeeze(np.array(predict_xyz).reshape([1, 9]))
    v_t_pos1 = np.abs(next_pt[:6] - actual_pos[:6])
    print('difference:',v_t_pos1-v_t_pos)
    gp_prediction = GP.get_GP_prediction(self,predict_xyz[:6])
    # gp_prediction = np.zeros(6)
    pos_angle, vel_angle = cal_angle(self, predict_xyz[:6].squeeze(), third_pt, gp_prediction)
    weight = 0.7
    [phi_d, theta_d, psi_d] = weight*pos_angle + (1-weight)*vel_angle
    '''tracking desired nominal state'''
    # a = next_pt[6:]
    # (phi_d, theta_d, psi_d) = a

    self.env.cur_angle = np.array([phi_d, theta_d, psi_d])

    '''QP 2 : Using CLF to approximate the desired angle'''
    '''v2(phi) = |phi-phi_d| ,v2(theta) = |theta-theta_d|, v2(psi) = |psi-psi_d|'''


    G_angle = np.concatenate([np.concatenate([
                                        g[6,:].reshape([1,-1]),
                                        -g[6, :].reshape([1, -1]),
                                        g[7,:].reshape([1,-1]),
                                        -g[7, :].reshape([1, -1]),
                                        g[8,:].reshape([1,-1]),
                                        -g[8, :].reshape([1, -1]),
                                        # 2*((obs[0] - cur_pt[0])*g[3, :]/self.x_max**2 - (obs[1] - cur_pt[1])*g[4, :]/self.y_max **2
                                        # - (obs[2] - cur_pt[2])*g[5, :]/self.z_max**2),
                                        np.eye(M), -np.eye(M)], axis=0),
                        np.concatenate([-np.ones([6, 1]), np.zeros([2 * M, 1])], axis=0),
                        np.concatenate([np.zeros([6, 1]), -np.ones([2 * M, 1])], axis=0),
                        ], axis=1)
    h_angle = np.concatenate([
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[0] - f[6] + phi_d - np.dot(g[6,:].reshape([1,M]), u_rl)),
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[0] + f[6] - phi_d + np.dot(g[6,:].reshape([1,M]), u_rl)),
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[1] - f[7] + theta_d - np.dot(g[7,:].reshape([1,M]), u_rl)),
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[1] + f[7] - theta_d + np.dot(g[7,:].reshape([1,M]), u_rl)),
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[2] - f[8] + psi_d - np.dot(g[8,:].reshape([1,M]), u_rl)),
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[2] + f[8] - psi_d + np.dot(g[8,:].reshape([1,M]), u_rl)),
                        # (obs[0] - cur_pt[0]) * f[3, :] / self.x_max ** 2 - (obs[1] - cur_pt[1]) * f[4, :] / self.y_max ** 2
                        # - (obs[2] - cur_pt[2]) * f[5, :] / self.z_max ** 2 + gamma_pos_cbf*h_x,

                        -u_rl + up_b,
                        u_rl - low_b], axis=0)

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
    # a = np.squeeze(u_bar)
    # print(u_bar[1])


    return np.expand_dims(np.array(u_bar[:4]), 0),np.sum(v_t_pos),np.sum(v_t_angle), [phi_d, theta_d, psi_d]
