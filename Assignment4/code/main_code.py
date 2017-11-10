import numpy as np
import pdb
import re
import matplotlib.pyplot as plt
# from pylatex import Math
# from pylatex import Matrix
import argparse


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
        # assert curr_state >= 0
        if curr_state >= 0:
            return np.dot(self.weights, self.feat_mat[curr_state, :])
        else:
            return 0
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
    def __init__(self, mdp, N, lamb=0):
        self.mdp = mdp
        self.lr = 0.001
        self.N = N
        self.lamb = lamb
        self.gamma = self.mdp.gamma
        # self.exp_type = exp_type
        self.str_f = '{:.6} {:.6} {:.6} {:.6} {:.6} {:.6}\n'

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
        str_to_write = ''
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
                delta = self.gamma * vf_next - vf_curr
                tmp1 = self.gamma * self.lamb
                tmp2 = np.dot(elig_trace, fv_curr_state)
                tmp3 = (1 - self.lr * self.gamma * self.lamb * tmp2)
                elig_trace = tmp1 * elig_trace + tmp3 * fv_curr_state
                self.mdp.weights = (self.mdp.weights + self.lr * (delta + vf_curr - vf_prev) *
                                    elig_trace - self.lr * (vf_curr - vf_prev) * fv_curr_state)
                vf = self.mdp.get_whole_val_fn()
                str_to_write += self.str_f.format(*vf)
                vf_prev = vf_curr
                curr_state = next_state
                fv_curr_state = self.mdp.get_feat_vec(next_state)
                nit += 1
                tit += 1
            ep_no += 1
            print('ep_no', ep_no, 'tot_num_iterations', tit)

        str_lamb = str(float(self.lamb))
        fname = 'exp2_' + str_lamb[0] + '_' + str_lamb[-1] + '.txt'
        with open(fname, 'w') as f:
            f.write(str_to_write)


def graph_plot():
    with open('./exp1.txt') as f:
        lines = f.read()

    vf_array1 = np.array(pat.findall(lines)).astype(np.float)
    x_ax = np.arange(vf_array1.shape[0])
    # vf_array2 = np.zeros(vf_array1.shape)
    # vf_array2[vf_array1 < -1] = -np.log10(-vf_array1[vf_array1 < -1])
    # vf_array2[vf_array1 > 1] = np.log10(vf_array1[vf_array1 > 1])
    # vf_array2[np.abs(vf_array1) < 1] = 0
    vf_array2 = vf_array1
    plt.plot(x_ax, vf_array2[:, 0], label='state 1')
    plt.plot(x_ax, vf_array2[:, 1], label='state 2')
    plt.plot(x_ax, vf_array2[:, 2], label='state 3')
    plt.plot(x_ax, vf_array2[:, 3], label='state 4')
    plt.plot(x_ax, vf_array2[:, 4], label='state 5')
    plt.plot(x_ax, vf_array2[:, 5], label='state 6')
    # plt.xlabel('Number of TD(0) updates')
    # plt.ylabel('Value Function of all states')
    # plt.ylim(-1e31, 1e31)
    plt.yscale('symlog')
    plt.legend(bbox_to_anchor=(.05, 1), loc=2, borderaxespad=0.)
    # pdb.set_trace()
    plt.show()
    # plt.savefig('exp1.png')


def read_arr(fname):
    with open(fname) as f:
        lines = f.read()
    vf_array = np.array(pat.findall(lines)).astype(np.float)
    # pdb.set_trace()
    return np.mean(vf_array, axis=1)


def graph_plot2():
    vf_array00 = read_arr('./exp2_0_0.txt')
    vf_array02 = read_arr('./exp2_0_2.txt')
    vf_array04 = read_arr('./exp2_0_4.txt')
    vf_array06 = read_arr('./exp2_0_6.txt')
    vf_array08 = read_arr('./exp2_0_8.txt')
    vf_array10 = read_arr('./exp2_1_0.txt')

    x_ax = np.arange(vf_array00.shape[0])

    plt.plot(x_ax, vf_array00, label='lambda=0')
    plt.plot(x_ax, vf_array02, label='lambda=0.2')
    plt.plot(x_ax, vf_array04, label='lambda=0.4')
    plt.plot(x_ax, vf_array06, label='lambda=0.6')
    plt.plot(x_ax, vf_array08, label='lambda=0.8')
    plt.plot(x_ax, vf_array10, label='lambda=1')
    # plt.ylim(-1e31, 1e31)
    # plt.yscale('symlog')
    plt.legend(bbox_to_anchor=(.85, 1), loc=2, borderaxespad=0.)
    plt.show()
    # plt.savefig('exp2.png')


if __name__ == '__main__':
    # weights = np.array([1, 1, 1, 1, 1, 10, 1])
    # weights = np.array([1, 1, 1, 1, 1, 1, 1])
    # weights = np.array([-1.2, 1, 14, 1, 1, 10, 1])
    # weights = np.array([0, 0, 0, 0, 0, 0, 0])
    # N = 1000000
    pat = re.compile(r'([-\d.+e]*)\s([-\d.+e]*)\s([-\d.+e]*)\s([-\d.+e]*)\s([-\d.+e]*)\s([-\d.+e]*)\n')
    # N = 5000
    # lamb = 0
    # lamb = 0.2
    # lamb = 0.4
    # lamb = 0.6
    # lamb = 0.8
    # lamb = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=int, help="experiment")
    parser.add_argument('N', type=int, help='N')
    parser.add_argument('l', type=int, help='lamb')
    parser.add_argument('w1', type=float, help='w1')
    parser.add_argument('w2', type=float, help='w2')
    parser.add_argument('w3', type=float, help='w3')
    parser.add_argument('w4', type=float, help='w4')
    parser.add_argument('w5', type=float, help='w5')
    parser.add_argument('w6', type=float, help='w6')
    parser.add_argument('w7', type=float, help='w7')

    args = parser.parse_args()

    exp = args.exp
    assert exp == 1 or exp == 2
    N = args.N
    lamb = args.l
    assert lamb >= 0 and lamb <= 1
    weights = np.zeros(7)
    weights[0] = args.w1
    weights[1] = args.w2
    weights[2] = args.w3
    weights[3] = args.w4
    weights[4] = args.w5
    weights[5] = args.w6
    weights[6] = args.w7

    # print(exp, N, lamb, weights)

    mdp_inst = mdp_baird(weights)
    td_0 = td_lambda(mdp_inst, N, lamb)
    if exp == 1:
        td_0.train_model_exp1()
    elif exp == 2:
        td_0.train_model2()
    # td_0.train_model()
    # td_0.train_model1()
    # td_0.train_model_exp1()     #
    # td_0.train_model2()
    #
    # graph_plot()
