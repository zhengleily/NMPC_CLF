#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy

class Quad():

    def __init__(self, pos_x, pos_y, pos_z, vel_x,  vel_y, vel_z,
                 phi, theta, psi):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_z = vel_z
        self.phi = phi
        self.theta = theta
        self.psi = psi

        self.dt= 0.02
        self.m = 0.04
        self.l = 0.2
        self.g = 9.81
        # self.k1 = np.array([0.02, 0.02, 0.02])
        self.k1 = np.array([0, 0, 0])
        # self.k2 = np.array([0.1, 0.1, 0.1])
        self.k2 = np.array([0, 0, 0])

    def next_state(self, state, act, cur_t, act_noise=0.1):
        '''
        Dynamics for quad
        '''

        act = act.reshape([4,])
        state = state.reshape([9,])
        # if act_noise:
        #     act = np.random.normal(act, act_noise)
        thrust = act[0]
        omega = act[1:]
        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, phi, theta, psi] = state
        # wind = self.wind(pos_x, pos_y, pos_z)
        wind = self.wind(pos_x, pos_y, pos_z, cur_t)
        Rotation = np.array([
            np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi),
            np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi),
            np.cos(theta) * np.cos(phi)
        ])
        omega2euler = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin((phi))],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        accel = np.array([0,0,-self.g]) + (Rotation * thrust + wind - self.k1 * np.square(np.array([vel_x,vel_y,vel_z]))) / self.m
        vel = np.array([vel_x,vel_y,vel_z]) + accel * self.dt
        angle_dot = np.dot(omega2euler,omega).squeeze()

        return np.squeeze(np.concatenate([vel,accel,angle_dot], axis=0))

    def getState(self):
        return np.array(copy.deepcopy([self.pos_x, self.pos_y, self.pos_z,
        self.vel_x, self.vel_y, self.vel_z,
        self.phi, self.theta, self.psi]))

    # Pick wind speed function.
    def wind(self, x, y, z, t):
        v = [
            0*1.75,
            0*1.75,
            0*0.05,
        ]
        return np.array(v)
    #
    # def wind(self, x, y, z, t):
    #     if t >= 600:
    #         v = [
    #             -0.3 + 0.2 * np.cos((x - 1) * 0.86 / 20) * 0.86 / 1 + 0.5,
    #             0.3 - 0.2 * np.cos((y - 5) * 0.86 / 20) * 0.86 / 1 + 0.5,
    #             0.1 + 0.3,
    #         ]
    #     if t >= 300 and t <= 600:
    #         v = [
    #             -0.3 + 0.2 * np.cos((x - 1) * 0.86 / 20) * 0.86 / 1,
    #             0.3 - 0.2 * np.cos((y - 5) * 0.86 / 20) * 0.86 / 1,
    #             0.1,
    #         ]
    #     else:
    #         v = [
    #             -0.30,
    #             0.30,
    #             0.1,
    #         ]
    #     return np.array(v)

    def set_state(self, state):
        [self.pos_x, self.pos_y, self.pos_z,
         self.vel_x, self.vel_y, self.vel_z,
         self.phi, self.theta, self.psi] = state


class quad_env():
    def __init__(self):
        # set params for quad
        self.L = 20
        self.t = 0
        self.g = 9.81
        self.dt= 0.02
        self.state_dim = 9
        self.action_dim = 4

        # self.curve_pos = lambda t: np.array([-2 * np.sin(0.5 * t), 5 + 2 * np.cos(0.5 * t), 0 * t + 1])
        # self.curve_vel = lambda t: np.array([-np.cos(0.5 * t), -np.sin(0.5 * t), 0 * t])
        # self.curve_accel_0 = np.array([-1,0, 0])

        self.curve_pos = lambda t: np.array([
                    np.clip(1.5 * (t - 0.02 * 200), 0, 1.5 * 0.02 * 100)
                    + np.clip(1.5 * (t - 0.02 * 500), 0, 1.5 * 0.02 * 100)
                    + np.clip(1.5 * (t - 0.02 * 800), 0, 1.5 * 0.02 * 200),

                    np.clip(1.5 * t, 0, 1.5 * 0.02 * 200)
                    + np.clip(-1.5 * (t - 0.02 * 300), -1.5 * 0.02 * 200, 0)
                    + np.clip(1.5 * (t - 0.02 * 600), 0, 1.5 * 0.02 * 200),

                    np.clip(1.5 * t, 0, 1.5 * 0.02 * 200)
                    + np.clip(-1.5 * (t - 0.02 * 300), -1.5 * 0.02 * 200, 0)
                    + np.clip(1.5 * (t - 0.02 * 600), 0, 1.5 * 0.02 * 200)])
        self.curve_vel = lambda t: np.array([
                    ((np.clip(1.5 * (t + 0.02 - 0.02 * 200), 0, 1.5 * 0.02 * 100)
                      + np.clip(1.5 * (t + 0.02 - 0.02 * 500), 0, 1.5 * 0.02 * 100)
                      + np.clip(1.5 * (t + self.dt- 0.02 * 800), 0, 1.5 * 0.02 * 200))
                     - (np.clip(1.5 * (t - 0.02 * 200), 0, 1.5 * 0.02 * 100)
                        + np.clip(1.5 * (t - 0.02 * 500), 0, 1.5 * 0.02 * 100)
                        + np.clip(1.5 * (t - 0.02 * 800), 0, 1.5 * 0.02 * 200))) / self.dt,

                    ((np.clip(1.5 * (t + self.dt), 0, 1.5 * 0.02 * 200)
                      + np.clip(-1.5 * (t + self.dt - 0.02 * 300), -1.5 * 0.02 * 200, 0)
                      + np.clip(1.5 * (t + self.dt - 0.02 * 600), 0, 1.5 * 0.02 * 200))
                     - (np.clip(1.5 * t, 0, 1.5 * 0.02 * 200)
                        + np.clip(-1.5 * (t - 0.02 * 300), -1.5 * 0.02 * 200, 0)
                        + np.clip(1.5 * (t - 0.02 * 600), 0, 1.5 * 0.02 * 200))) / self.dt,

                    ((np.clip(1.5 * (t + self.dt), 0, 1.5 * 0.02 * 200)
                      + np.clip(-1.5 * (t + self.dt - 0.02 * 300), -1.5 * 0.02 * 200, 0)
                      + np.clip(1.5 * (t + self.dt - 0.02 * 600), 0, 1.5 * 0.02 * 200))
                     - (np.clip(1.5 * t, 0, 1.5 * 0.02 * 200)
                        + np.clip(-1.5 * (t - 0.02 * 300), -1.5 * 0.02 * 200, 0)
                        + np.clip(1.5 * (t - 0.02 * 600), 0, 1.5 * 0.02 * 200))) / self.dt,
                ])

        self.curve_accel_0 = np.array([0, 0.5, 0.5])
        #
        # self.curve_pos = lambda t: np.array([0,0,4])
        # self.curve_vel = lambda t: np.array([0,0,0])
        # self.curve_accel_0 = np.array([0, 0, 0])
        self.init_pos = self.curve_pos(0)
        self.init_vel = self.curve_vel(0)
        self.init_accel =  self.curve_accel_0
        self.psi_d = 0
        self.init_theta,self.init_phi = self.cal_theta_phi(self.init_accel, self.psi_d)
        self.quad = Quad(self.init_pos[0], self.init_pos[1], self.init_pos[2],
                         self.init_vel[0], self.init_vel[1], self.init_vel[2],
                         self.init_phi, self.init_theta, self.psi_d)
        # self.quad = Quad(0,0,0,0,0,0,0,0,0)
        # self.cur_angle = np.zeros(3)
        # self.curve = np.load('curve.npy')
        self.cur_angle = np.array([self.init_phi, self.init_theta, self.psi_d])
        self.barrier = np.array([
        #     [3, 3, 3, 1],
        #     [7, 7, 7, 1],
        #     [3, 5, 1, 1],
        #     [6, 6, 4, 1],
        #     [7, 3, 5, 1],
        #     [2, 6, 8, 1],
        #     [4, 4, 2, 1],
        #     [5, 7, 5, 1],
        #     [7, 7, 5, 1],
        [-2 * np.sin(0.5 * 620 * 0.02), 5 + 2 * np.cos(0.5 * 620 * 0.02), .4 * 620 * 0.02],
        [-2 * np.sin(0.5 * 520 * 0.02) - 0.3, 4.15 + 2 * np.cos(0.5 * 600 * 0.02), .4 * 580 * 0.02],
        [-2 * np.sin(0.5 * 520 * 0.02) - 0.35, 4.6 + 2 * np.cos(0.5 * 600 * 0.02), .4 * 590 * 0.02],
        [-2 * np.sin(0.5 * 600 * 0.02), 5 + 2 * np.cos(0.5 * 610 * 0.02), .4 * 520 * 0.02]
            # self.curve_pos(20),
        ])
        self.obstacles = []
        self.obstacle_without_center = []
        # for i in range(10):
        #     center = np.random.random(3,) * np.array([4,4,4]) + np.array([-2,3,0])
        #     scale = 1
        #     pts = np.random.random([3, 6])
        #     pts = pts - np.mean(pts, axis=1)[:, np.newaxis]
        #     pts = scale * pts + center[:, np.newaxis]
        #     self.obstacles.append(pts)
        # for obstacle in self.barrier:
        #     center = obstacle[:3]
        #     scale = 1
        #     # pts = np.array([[1/3,0,0],[0,-1/3/3**0.5,0],[0,1/3/3**0.5,0],[1/9,0,4]]).T
        #     pts = np.random.random([3, 6])
        #     pts = pts - np.mean(pts, axis=1)[:, np.newaxis]
        #     self.obstacle_without_center.append(pts)
        #     pts = scale * pts + center[:, np.newaxis]
        #     self.obstacles.append(pts)

    def cal_theta_phi(self, accel, psi_d):
        x_dd, y_dd, z_dd = accel
        belta_a = x_dd * np.cos(psi_d) + y_dd * np.sin(psi_d)
        belta_b = z_dd + self.g
        belta_c = x_dd * np.sin(psi_d) - y_dd * np.cos(psi_d)
        theta_d = np.arctan2(belta_a,belta_b)
        phi_d = np.arctan2(belta_c, np.sqrt(belta_a ** 2 + belta_b ** 2))
        return theta_d, phi_d

    def getReward(self, action):
        s = self.quad.getState()
        r = np.sum(abs(action),axis=0)
        return r

    def reset(self, state=None):
        self.t = 0
        # [a,b,c] = np.random.random([3,])
        # a,b,c = 0.25, 0.25, 0.25
        # self.curve_pos = lambda t: np.array([(a+b)*t,(c+b)*t,(a+c)*t])
        # self.curve_vel = lambda t: np.array([(a+b),(c+b),(a+c)])
        # self.curve_accel = lambda t: np.array([0, 0, 0])
        self.init_pos = self.curve_pos(0)
        self.init_vel = self.curve_vel(0)
        self.init_accel = self.curve_accel_0
        self.psi_d = 0
        self.init_theta,self.init_phi = self.cal_theta_phi(self.init_accel, self.psi_d)
        self.cur_angle = np.array([self.init_phi, self.init_theta, self.psi_d])
        if not state:
            self.quad.set_state([self.init_pos[0]+0., self.init_pos[1]+0., self.init_pos[2]-0.,
                                self.init_vel[0],self.init_vel[1],self.init_vel[2],0, 0,0])
            # self.quad.set_state([self.init_pos[0]+0., self.init_pos[1]+0., self.init_pos[2]-0.,
            #                       0,0,0,0,0,0])
        else:
            state = np.squeeze(state)
            self.quad.set_state(state)
        return self.quad.getState()

    def step(self, action):
        # Runge-Kutta methods
        state = self.quad.getState()
        k1 = np.array(self.quad.next_state(state, action, self.t, 0) * self.quad.dt)
        # k2 = np.array(self.quad.next_state(state + k1/2.0, action) * self.quad.dt)
        # k3 = np.array(self.quad.next_state(state + k2/2.0, action) * self.quad.dt)
        # k4 = np.array(self.quad.next_state(state + k3, action) * self.quad.dt)
        r = self.getReward(action)
        self.t = self.t + self.quad.dt
        flag = False
        self.quad.set_state(state + (k1))# + 2.0 * (k2 + k3) + k4) / 6.0)

        # self.obstacles[0] = self.obstacle_without_center[0] + self.curve_pos(max(0, 4 - self.t)).reshape(-1,1)
        # self.obstacles[1] = self.obstacle_without_center[1] + self.curve_pos(max(0, 12 - self.t)).reshape(-1,1)
        # self.obstacles[2] = self.obstacle_without_center[2] + self.curve_pos(max(0, 20 - self.t)).reshape(-1,1)

        if self.t == self.L:
            self.t = 0
            flag = True
        return self.quad.getState(), r, flag

    def predict_f_g(self, obs):
        # Params
        self.dt = self.quad.self.dt
        m = self.quad.m
        g = self.quad.g
        dO = self.state_dim
        obs = obs.reshape(-1, dO)

        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, phi, theta, psi] = obs.T

        sample_num = obs.shape[0]
        # calculate f with size [-1,9,1]
        f = np.concatenate([
            np.array(vel_x).reshape(sample_num, 1),
            np.array(vel_y).reshape(sample_num, 1),
            np.array(vel_z - g * self.dt).reshape(sample_num, 1),
            np.zeros([sample_num, 1]),
            np.zeros([sample_num, 1]),
            -g * np.ones([sample_num, 1]).reshape(-1, 1),
            np.zeros([sample_num,3]),
        ], axis=1)
        f = f * self.dt + obs
        f = f.reshape([-1, dO, 1])
        # calculate g with size [-1,9,4]
        accel_x = np.concatenate([
            (np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)).reshape([-1, 1, 1]) / m,
            np.zeros([sample_num, 1, 3])
        ], axis=2)
        accel_y = np.concatenate([
            (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)).reshape([-1, 1, 1]) / m,
            np.zeros([sample_num, 1, 3])
        ], axis=2)
        accel_z = np.concatenate([
            (np.cos(theta) * np.cos(phi)).reshape([-1, 1, 1]) / m,
            np.zeros([sample_num, 1, 3])
        ], axis=2)
        phi_dot = np.concatenate([
            np.zeros([sample_num, 1, 1]),
            np.ones([sample_num, 1, 1]),
            (np.sin(phi)*np.tan(theta)).reshape([-1,1,1]),
            (np.cos(phi)*np.tan(theta)).reshape([-1,1,1])
        ], axis=2)
        theta_dot = np.concatenate([
            np.zeros([sample_num, 1, 2]),
            np.cos(phi).reshape([-1,1,1]),
            -np.sin(phi).reshape([-1,1,1])
        ], axis=2)
        psi_dot = np.concatenate([
            np.zeros([sample_num, 1, 2]),
            (np.sin(phi)/np.cos(theta)).reshape([-1,1,1]),
            (np.cos(phi)/np.cos(theta)).reshape([-1,1,1])
        ], axis=2)
        g_mat = np.concatenate([
            accel_x * self.dt,
            accel_y * self.dt,
            accel_z * self.dt,
            accel_x,
            accel_y,
            accel_z,
            phi_dot,
            theta_dot,
            psi_dot
        ], axis=1)
        g_mat = self.dt * g_mat
        return f, g_mat, np.copy(obs)

    def _predict_next_obs_uncertainty(self, obs, cur_acs, t=None, agent=None):
        f_nom, g, x = self.predict_f_g(obs)
        cur_acs = np.asarray(cur_acs).reshape([-1, 4, 1])

        # if (agent.firstIter == 1 and t < 20 * 0.02):
        #     [m1, std1] = agent.GP_model[0].predict(x, return_std=True)
        #     [m2, std2] = agent.GP_model[1].predict(x, return_std=True)
        #     [m3, std3] = agent.GP_model[2].predict(x, return_std=True)
        #     [m4, std4] = agent.GP_model[3].predict(x, return_std=True)
        #     [m5, std5] = agent.GP_model[4].predict(x, return_std=True)
        #     [m6, std6] = agent.GP_model[5].predict(x, return_std=True)
        #
        #     f_nom[:, 6] = f_nom[:, 6] + m1 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * 0
        #     f_nom[:, 7] = f_nom[:, 7] + m2 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * 0
        #     f_nom[:, 9] = f_nom[:, 9] + m3 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * 0
        #     f_nom[:, 10] = f_nom[:, 10] + m4 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * 0
        #     f_nom[:, 12] = f_nom[:, 12] + m5 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * 0
        #     f_nom[:, 13] = f_nom[:, 13] + m6 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * 0
        # elif (agent.firstIter == 1 and t >= 20 * 0.02):
        #     [m1, std1] = agent.GP_model[0].predict(x, return_std=True)
        #     [m2, std2] = agent.GP_model[1].predict(x, return_std=True)
        #     [m3, std3] = agent.GP_model[2].predict(x, return_std=True)
        #     [m4, std4] = agent.GP_model[3].predict(x, return_std=True)
        #     [m5, std5] = agent.GP_model[4].predict(x, return_std=True)
        #     [m6, std6] = agent.GP_model[5].predict(x, return_std=True)
        #
        #     f_nom[:, 6] = f_nom[:, 6] + m1 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2))
        #     f_nom[:, 7] = f_nom[:, 7] + m2 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2))
        #     f_nom[:, 9] = f_nom[:, 9] + m3 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2))
        #     f_nom[:, 10] = f_nom[:, 10] + m4 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2))
        #     f_nom[:, 12] = f_nom[:, 12] + m5 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2))
        #     f_nom[:, 13] = f_nom[:, 13] + m6 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2))
        # else:
        #     [m1, std1] = agent.GP_model_prev[0].predict(x, return_std=True)
        #     [m2, std2] = agent.GP_model_prev[1].predict(x, return_std=True)
        #     [m3, std3] = agent.GP_model_prev[2].predict(x, return_std=True)
        #     [m4, std4] = agent.GP_model_prev[3].predict(x, return_std=True)
        #     [m5, std5] = agent.GP_model_prev[4].predict(x, return_std=True)
        #     [m6, std6] = agent.GP_model_prev[5].predict(x, return_std=True)
        #
        #     f_nom[:, 6] = f_nom[:, 6] + m1 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * std1
        #     f_nom[:, 7] = f_nom[:, 7] + m2 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * std2
        #     f_nom[:, 9] = f_nom[:, 9] + m3 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * std3
        #     f_nom[:, 10] = f_nom[:, 10] + m4 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * std4
        #     f_nom[:, 12] = f_nom[:, 12] + m5 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * std5
        #     f_nom[:, 13] = f_nom[:, 13] + m6 + np.sqrt(2 * np.log(1.667 * 15 * (np.pi * step_time) ** 2)) * std6

        # f = f_nom.reshape(-1, 1, 9)
        next_obs = f_nom + np.matmul(g, cur_acs)
        return next_obs

    def cal_u1(self,state,state_dsr):
        vector = self.direction(state,state_dsr)[0]
        u1 = np.linalg.norm(vector)
        return u1

    def direction(self,state,state_dsr,gp_prediction=np.zeros(6,)):
        self.dt = self.quad.self.dt
        g = self.quad.g
        m = self.quad.m
        pos_direction = ((state_dsr[:3] - state[:3] - gp_prediction[:3] - state[3:6] * self.dt) / self.dt ** 2 + np.array([0, 0, g])) * m
        vel_direction = ((state_dsr[3:6] - gp_prediction[3:6] - state[3:6]) / self.dt + np.array([0, 0, g])) * m
        return pos_direction, vel_direction

    def cost_func(self, action, state, plan_hor, dO, dU, t0, agent=None):
        state = np.asarray(state).reshape((dO, ))
        action = np.asarray(action).reshape((-1, plan_hor, dU))
        t = self.t
        init_obs = np.tile(state, (action.shape[0],1))
        init_cost = np.zeros((action.shape[0],))
        self.dt = self.quad.self.dt
        # t0 = int(t0//self.dt)
        # plan_hor = min(plan_hor, self.curve.shape[0] - t0)
        for i in range(plan_hor):
            cur_acs = action[:, i, :].reshape(-1, dU, 1)
            next_obs = self._predict_next_obs_uncertainty(init_obs, cur_acs, t, agent)
            init_obs = np.squeeze(next_obs)
            # Here to set the tracking curve
            # init_cost += np.sum(np.square(target[3:6].T - init_obs[:,3:6]), axis=1) * np.exp(-i*0.01)

            target = self.curve_pos(t0+i*self.dt)
            # target_v = self.curve_vel(t0+i*self.dt)
            init_cost += np.sum(np.square(target[:3] - init_obs[:, :3]), axis=1)
            # init_cost += np.sum(np.square(init_obs[:, 8]))
            # for b in self.barrier:
            #     d = np.sqrt(np.sum(np.square(init_obs[:, :3] - b[:3]), axis=1))
            #     temp = (np.sqrt(np.sum(np.square(init_obs[:, :3] - b[:3]), axis=1)) <= np.sqrt(0.3) + self.quad.l)
            #     init_cost += temp*1000
            # init_cost + (init_obs[:, 2] <= 0) * 0.1
        cost = init_cost
        return cost
