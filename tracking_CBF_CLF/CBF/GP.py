#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from cvxopt import matrix
from cvxopt import solvers
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
class GP():

    def build_GP_model(self):
        N = 6  # 3 GPs for 3 states involved in CBF,x、y、z
        GP_list = []
        noise = 1e-10
        for i in range(N):
            kern = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kern, alpha=noise, n_restarts_optimizer=10)
            GP_list.append(gp)
        self.GP_model = GP_list
        self.train_X = []
        self.train_y1 = []
        self.train_y2 = []
        self.train_y3 = []
        self.train_y4 = []
        self.train_y5 = []
        self.train_y6 = []
        self.datas1 = []
        self.datas2 = []
        self.datas3 = []
        self.datas4 = []
        self.datas5 = []
        self.datas6 = []

    # Build GP dynamics model
    def update_GP_dynamics(self, path):
        N = self.observation_space
        X = path['Observation']
        U = path['Action'].reshape(-1,4)
        L = X.shape[0]
        err = np.zeros((L - 1, N))
        for i in range(L - 1):
            [f, g, x1] = self.env.predict_f_g(X[i, :])
            f = np.ravel(f)
            g = np.ravel(g)
            x1 = np.ravel(x1)
            # err[i, :] = X[i + 1, :] - f - np.matmul(g, np.square(U[i, :]))
            X_next = np.squeeze(self.env._predict_next_obs_uncertainty(X[i, :],U[i]),axis=2)
            err[i, :] = X[i + 1, :] - X_next
        S = X[0:L - 1, :6]
        t1 = time.time()
        self.GP_model[0].fit(S, err[:, 0])
        self.GP_model[1].fit(S, err[:, 1])
        self.GP_model[2].fit(S, err[:, 2])
        self.GP_model[3].fit(S, err[:, 3])
        self.GP_model[4].fit(S, err[:, 4])
        self.GP_model[5].fit(S, err[:, 5])
        t2 = time.time() - t1
        print('The GP update time:', t2)

    def get_GP_dynamics(self, obs, u_rl):
        s = obs.reshape(-1,9)
        [f_nom, g, x] = self.env.predict_f_g(obs)
        f_nom = np.ravel(f_nom)
        # g = np.ravel(g)
        x = np.ravel(x)[:6]
        f = np.copy(f_nom)

        [m0, std0] = self.GP_model[0].predict(x.reshape(1, -1), return_std=True)
        [m1, std1] = self.GP_model[1].predict(x.reshape(1, -1), return_std=True)
        [m2, std2] = self.GP_model[2].predict(x.reshape(1, -1), return_std=True)
        [m3, std3] = self.GP_model[3].predict(x.reshape(1, -1), return_std=True)
        [m4, std4] = self.GP_model[4].predict(x.reshape(1, -1), return_std=True)
        [m5, std5] = self.GP_model[5].predict(x.reshape(1, -1), return_std=True)
        f[0] = f_nom[0] + m0
        f[1] = f_nom[1] + m1
        f[2] = f_nom[2] + m2
        f[3] = f_nom[3] + m3
        f[4] = f_nom[4] + m4
        f[5] = f_nom[5] + m5

        return [np.squeeze(f),np.array([m0,m1,m2,m3,m4,m5]), np.squeeze(g), np.squeeze(obs),  np.array([np.squeeze(std0),
                                                                       np.squeeze(std1),
                                                                       np.squeeze(std2),
                                                                       np.squeeze(std3),
                                                                       np.squeeze(std4),
                                                                       np.squeeze(std5),
                                                                       0, 0, 0])]
    def get_GP_dynamics_prev(self, obs, u_rl):
        s = obs.reshape(-1,9)
        [f_nom, g, x] = self.env.predict_f_g(obs)
        f_nom = np.ravel(f_nom)
        g = np.ravel(g)
        x = np.ravel(x)[:6]
        f = f_nom
        [m0, std0] = self.GP_model_prev[0].predict(x.reshape(1, -1), return_std=True)
        [m1, std1] = self.GP_model_prev[1].predict(x.reshape(1, -1), return_std=True)
        [m2, std2] = self.GP_model_prev[2].predict(x.reshape(1, -1), return_std=True)
        [m3, std3] = self.GP_model_prev[3].predict(x.reshape(1, -1), return_std=True)
        [m4, std4] = self.GP_model_prev[4].predict(x.reshape(1, -1), return_std=True)
        [m5, std5] = self.GP_model_prev[5].predict(x.reshape(1, -1), return_std=True)
        f[0] = f_nom[0] + m0
        f[1] = f_nom[1] + m1
        f[2] = f_nom[2] + m2
        f[3] = f_nom[3] + m3
        f[4] = f_nom[4] + m4
        f[5] = f_nom[5] + m5

        return [np.squeeze(f), np.squeeze(g), np.squeeze(obs), np.array([np.squeeze(std0),
                                                                       np.squeeze(std1),
                                                                       np.squeeze(std2),
                                                                       np.squeeze(std3),
                                                                       np.squeeze(std4),
                                                                       np.squeeze(std5),
                                                                       0, 0, 0])]
    def get_GP_prediction(self, obs):
        x = obs[:6]
        m0 = self.GP_model[0].predict(x.reshape(1, -1), return_std=False)
        m1 = self.GP_model[1].predict(x.reshape(1, -1), return_std=False)
        m2 = self.GP_model[2].predict(x.reshape(1, -1), return_std=False)
        m3 = self.GP_model[3].predict(x.reshape(1, -1), return_std=False)
        m4 = self.GP_model[4].predict(x.reshape(1, -1), return_std=False)
        m5 = self.GP_model[5].predict(x.reshape(1, -1), return_std=False)
        return np.array([m0,m1,m2,m3,m4,m5,0,0,0])