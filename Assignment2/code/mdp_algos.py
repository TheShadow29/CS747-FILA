import numpy as np


class mdp_solver(object):
    def __init__(self, S, A, reward_mat, trans_mat, _gamma,  _batch_size, _random_seed):
        self.tot_states_num = S
        self.tot_action_num = A
        self.reward_matrix = reward_mat
        self.trans_matrix = trans_mat
        self.gamma = _gamma
        self.batch_size = _batch_size
        self.rand_seed = _random_seed
        return

    # def linear_programming(self):
        #
