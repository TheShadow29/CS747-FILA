import numpy as np


class mdp_baird(object):
    def __init__(self, weights):
        self.weights = weights
        # self.val_mat = np.zeros((6, self.weights.shape[0]))
        self.val_mat = np.array([[2, 0, 0, 0, 0, 0, 1],
                                 [0, 2, 0, 0, 0, 0, 1],
                                 [0, 0, 2, 0, 0, 0, 1],
                                 [0, 0, 0, 2, 0, 0, 1],
                                 [0, 0, 0, 0, 2, 0, 1],
                                 [0, 0, 0, 0, 0, 1, 2]])
        self.num_states = self.val_mat.shape[0]
        self.gamma = 0.99
        self.start_state = 0
        self.state_counter = 0

    def get_init_state(self):
        self.state_counter += 1
        if self.state_counter == self.num_states:
            self.state_counter = 0
        return self.state_counter % self.num_states

    def reward_fn(self, s, a, s_next):
        return 0

    def get_next_state(self, s, a):
        if s < 5:
            return 5
        elif s == 5:
            if np.random.random() < 0.99:
                return 6
            else:
                return 5
        elif s == 6:
            return -1


class td_zero(object):
    def __init__(self, mdp):
        self.mdp = mdp
        self.lr = 0.001

    # def update(self):


if __name__ == '__main__':
    weights = [0, 1, 1, 1, 1, 1, 10, 1]
    mdp_inst = mdp_baird(weights)
