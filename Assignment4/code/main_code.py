import numpy as np
import pdb
import re
import matplotlib.pyplot as plt

class mdp_baird(object):
    def __init__(self, weights):
        self.weights = weights
        # self.val_mat = np.zeros((6, self.weights.shape[0]))
        self.feat_mat = np.array([[2, 0, 0, 0, 0, 0, 1],
                                 [0, 2, 0, 0, 0, 0, 1],
                                 [0, 0, 2, 0, 0, 0, 1],
                                 [0, 0, 0, 2, 0, 0, 1],
                                 [0, 0, 0, 0, 2, 0, 1],
                                 [0, 0, 0, 0, 0, 1, 2]])
        self.num_states = self.feat_mat.shape[0]
        self.gamma = 0.99
        self.start_state = 0
        self.state_counter = 0

    def get_init_state(self):
        self.state_counter += 1
        if self.state_counter == self.num_states:
            self.state_counter = 0
        return self.state_counter % self.num_states

    def get_init_state2(self):
        return np.random.randint(0, self.num_states - 1)

    def reward_fn(self, s, s_next):
        return 0

    def get_next_state(self, s):
        if s < 5:
            return 5
        elif s == 5:
            if np.random.random() > 0.99:
                return -1
            else:
                return 5
        # elif s == 6:
        #     return -1

    def get_out_val_fn(self, curr_state):
        # pdb.set_trace()
        return np.dot(self.weights, self.feat_mat[curr_state, :])
        # try:
        # return np.dot(self.weights, self.feat_mat[curr_state, :])
        # except Exception as e:
        # pdb.set_trace()

    def get_whole_val_fn(self):
        return np.dot(self.feat_mat, self.weights)

    def get_feat_vec(self, curr_state):
        return self.feat_mat[curr_state, :]
# class td_zero(object):
#     def __init__(self, mdp):
#         self.mdp = mdp
#         self.lr = 0.001

# def update(self):


class td_lambda(object):
    def __init__(self, mdp, N, exp_type=1, lamb=0):
        self.mdp = mdp
        self.lr = 0.001
        self.N = N
        self.lamb = lamb
        self.gamma = self.mdp.gamma
        self.exp_type = exp_type
        self.str_f = '{:.6} {:.6} {:.6} {:.6} {:.6} {:.6}\n'

    def train_model(self):
        for ep in range(self.N):
            curr_state = self.mdp.get_init_state()
            fv_curr_state = self.mdp.get_feat_vec(curr_state)
            elig_trace = np.zeros(self.mdp.weights.shape)
            vf_prev = 0
            next_state = curr_state
            nit = 0
            while (next_state != -1):
                next_state = self.mdp.get_next_state(curr_state)
                vf_curr = self.mdp.get_out_val_fn(curr_state)
                vf_next = self.mdp.get_out_val_fn(next_state)
                delta = self.gamma * (vf_next - vf_curr)
                tmp1 = self.gamma * self.lamb
                tmp2 = np.dot(elig_trace, fv_curr_state)
                tmp3 = (1 - self.lr * self.gamma * self.lamb * tmp2)
                elig_trace = tmp1 * elig_trace + tmp3 * fv_curr_state
                self.mdp.weights = (self.mdp.weights + self.lr * (delta + vf_curr - vf_prev) *
                                    elig_trace - self.lr * (vf_curr - vf_prev) * fv_curr_state)
                vf_prev = vf_curr
                curr_state = next_state
                fv_curr_state = self.mdp.get_feat_vec(next_state)
                nit += 1

            print('ep_no', ep, 'num_iterations', nit)

    def train_model1(self):
        for ep in range(self.N):
            curr_state = self.mdp.get_init_state()
            fv_curr_state = self.mdp.get_feat_vec(curr_state)
            # vf_prev = 0
            next_state = curr_state
            nit = 0
            while (next_state != -1):
                next_state = self.mdp.get_next_state(curr_state)
                vf_curr = self.mdp.get_out_val_fn(curr_state)
                vf_next = self.mdp.get_out_val_fn(next_state)
                delta = self.gamma * (vf_next - vf_curr)
                self.mdp.weights = self.mdp.weights + self.lr * delta * fv_curr_state
                # vf_prev = vf_curr
                curr_state = next_state
                fv_curr_state = self.mdp.get_feat_vec(next_state)
                nit += 1

            print('ep_no', ep, 'num_iterations', nit)

    def train_model_exp1(self):
        str_to_write = ''
        for upd in range(self.N):
            curr_state = self.mdp.get_init_state()
            fv_curr_state = self.mdp.get_feat_vec(curr_state)
            # elig_trace = np.zeros(self.mdp.weights.shape)
            # vf_prev = 0
            # next_state = curr_state
            # nit = 0
            # while (next_state != -1):
            next_state = self.mdp.get_next_state(curr_state)
            vf_curr = self.mdp.get_out_val_fn(curr_state)
            vf_next = self.mdp.get_out_val_fn(next_state)
            delta = self.gamma * vf_next - vf_curr
            # pdb.set_trace()
            self.mdp.weights = self.mdp.weights + self.lr * delta * fv_curr_state
            vf = self.mdp.get_whole_val_fn()
            str_to_write += self.str_f.format(*vf)
            # vf_prev = vf_curr
            # curr_state = next_state
            # fv_curr_state = self.mdp.get_feat_vec(next_state)
            # nit += 1
            print('upd_no', upd)
        with open('exp1.txt', 'w') as f:
            f.write(str_to_write)

    def train_model2(self):
        tit = 0
        ep_no = 0
        while tit < self.N:
            curr_state = self.mdp.get_init_state()
            fv_curr_state = self.mdp.get_feat_vec(curr_state)
            elig_trace = np.zeros(self.mdp.weights.shape)
            vf_prev = 0
            next_state = curr_state
            nit = 0
            while (next_state != -1) and tit < self.N:
                next_state = self.mdp.get_next_state(curr_state)
                vf_curr = self.mdp.get_out_val_fn(curr_state)
                vf_next = self.mdp.get_out_val_fn(next_state)
                delta = self.gamma * (vf_next - vf_curr)
                tmp1 = self.gamma * self.lamb
                tmp2 = np.dot(elig_trace, fv_curr_state)
                tmp3 = (1 - self.lr * self.gamma * self.lamb * tmp2)
                elig_trace = tmp1 * elig_trace + tmp3 * fv_curr_state
                self.mdp.weights = (self.mdp.weights + self.lr * (delta + vf_curr - vf_prev) *
                                    elig_trace - self.lr * (vf_curr - vf_prev) * fv_curr_state)
                vf_prev = vf_curr
                curr_state = next_state
                fv_curr_state = self.mdp.get_feat_vec(next_state)
                nit += 1
                tit += 1
            ep_no += 1
            print('ep_no', ep_no, 'tot_num_iterations', tit)


def graph_plot():
    with open('./exp1.txt') as f:
        lines = f.read()

    pat = re.compile(r'([-\d.+e]*)\s([-\d.+e]*)\s([-\d.+e]*)\s([-\d.+e]*)\s([-\d.+e]*)\s([-\d.+e]*)\n')
    vf_array1 = np.array(pat.findall(lines)).astype(np.float)
    x_ax = np.arange(vf_array1.shape[0])
    vf_array2 = np.zeros(vf_array1.shape)
    vf_array2[vf_array1 < -1] = -np.log10(-vf_array1[vf_array1 < -1])
    vf_array2[vf_array1 > 1] = np.log10(vf_array1[vf_array1 > 1])
    vf_array2[np.abs(vf_array1) < 1] = 0
    plt.plot(x_ax, vf_array2[:, 0])
    plt.plot(x_ax, vf_array2[:, 1])
    plt.plot(x_ax, vf_array2[:, 2])
    plt.plot(x_ax, vf_array2[:, 3])
    plt.plot(x_ax, vf_array2[:, 4])
    plt.plot(x_ax, vf_array2[:, 5])
    plt.ylim(-40, 40)
    # pdb.set_trace()
    plt.show()


if __name__ == '__main__':
    # weights = np.array([1, 1, 1, 1, 1, 10, 1])
    weights = np.array([1, 1, 1, 1, 1, 1, 1])
    # weights = np.array([-1.2, 1, 14, 1, 1, 10, 1])
    # weights = np.array([0, 0, 0, 0, 0, 0, 0])
    N = 1000000
    # N = 5000
    lamb = 0
    # lamb = 0.2
    mdp_inst = mdp_baird(weights)
    td_0 = td_lambda(mdp_inst, N, lamb)
    # td_0.train_model()
    # td_0.train_model1()
    # td_0.train_model_exp1()     #
    # td_0.train_model2()
