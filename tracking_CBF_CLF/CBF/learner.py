import numpy as np
from .GP import GP
from .CBF import build_barrier
from barrier_comp import BARRIER

class LEARNER():
    def __init__(self, env, sess, optimize=True):
        self.firstIter = 1
        self.env = env
        self.action_bound_up = np.array([12,10,10,10])
        self.action_bound_low = np.array([0*9.81,-10,-10,-10])

        '''
        #Set up observation space and action space
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print('Observation space', self.observation_space)
        print('Action space', self.action_space)
        '''
        self.env = env
        # Set up observation space and action space
        self.observation_space = env.state_dim
        self.action_space = env.action_dim
        self.cost = env.cost_func
        # Build barrier function model
        build_barrier(self)
        # Build GP model of dynamics
        GP.build_GP_model(self)

        self.bar_comp = BARRIER(sess, 9, 4)

        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize
        self.safe_region = []
        self.former_datas1 = np.load('/home/zhenglei/Desktop/CEM_UAV/UAV_iris_random_obs/tracking/CBF/GP_DATAS/gp_datas1.npy')
        self.former_datas2 = np.load('/home/zhenglei/Desktop/CEM_UAV/UAV_iris_random_obs/tracking/CBF/GP_DATAS/gp_datas2.npy')
        self.former_datas3 = np.load('/home/zhenglei/Desktop/CEM_UAV/UAV_iris_random_obs/tracking/CBF/GP_DATAS/gp_datas3.npy')
        self.former_datas4 = np.load('/home/zhenglei/Desktop/CEM_UAV/UAV_iris_random_obs/tracking/CBF/GP_DATAS/gp_datas4.npy')
        self.former_datas5 = np.load('/home/zhenglei/Desktop/CEM_UAV/UAV_iris_random_obs/tracking/CBF/GP_DATAS/gp_datas5.npy')
        self.former_datas6 = np.load('/home/zhenglei/Desktop/CEM_UAV/UAV_iris_random_obs/tracking/CBF/GP_DATAS/gp_datas6.npy')

        print('Observation space', self.observation_space)
        print('Action space', self.action_space)

    def kernel(self, x1, x2):
        x1 = np.array(x1).reshape(-1, 6)
        x2 = np.array(x2).reshape(-1, 6)
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)

    def kernel_filter(self, x, gp_data):
        num_elites = 80
        total_data = np.concatenate((x, gp_data), axis=0)
        # similarity = self.kernel(total_data[:,:-1], current_state)
        similarity = self.kernel(x[-1, :-1], gp_data[:, :-1], )
        similarity1 = self.kernel(x[0, :-1], gp_data[0, :-1], )
        similarity2,similarity3 = x[0, :-1], gp_data[0, :-1]

        similarity = np.sum(similarity, axis=0).reshape(-1, 1)
        elites = gp_data[np.argsort(-similarity, axis=0)][:num_elites]
        elites = np.squeeze(elites)
        elites = np.concatenate((x, elites), axis=0)
        return np.squeeze(elites)

