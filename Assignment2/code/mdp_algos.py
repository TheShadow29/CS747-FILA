import numpy as np
# from pulp import *
import pulp
import pdb

class mdp_solver(object):
    def __init__(self, S, A, reward_mat, trans_mat, _gamma,  _batch_size, _random_seed):
        self.tot_states_num = S
        self.tot_action_num = A
        self.reward_matrix = reward_mat
        self.trans_matrix = trans_mat
        self.gamma = _gamma
        self.batch_size = _batch_size
        self.rand_seed = _random_seed
        self.out_format = '{} {}\n'
        return

    # def value_funciton(self, policy, curr_state):
    #     # Assume policy to be a vector with ind corresponding to action
    #     # curr_state is current index
    #     vf = 0
    #     for s_prime in range(self.tot_states_num):
    #         tmp1 = self.trans_matrix[curr_state, policy[curr_state], s_prime]
    #         tmp2 = self.reward_matrix[curr_state, policy[curr_state], s_prime]
    #         tmp3 = self.gamma * self.value_funciton(policy, s_prime)
    #         vf += tmp1 * (tmp2 + tmp3)
    #     return vf
    def value_func_to_policy(self, value_func):
        # policy = np.zeros(self.tot_states_num)
        q_star = np.zeros((self.tot_states_num, self.tot_action_num))
        for s in range(self.tot_states_num):
            for a in range(self.tot_action_num):
                q_star[s, a] = sum([self.trans_matrix[s, a, s_prime] *
                                    (self.reward_matrix[s, a, s_prime] +
                                     self.gamma * value_func[s_prime])
                                    for s_prime in range(self.tot_states_num)])
        policy = np.argmax(q_star, axis=1)
        return policy

    def linear_programming(self):
        prob = pulp.LpProblem('mdp_lp', pulp.LpMinimize)
        states = np.arange(0, self.tot_states_num, dtype=int)
        v_star = pulp.LpVariable.dicts('v_star', states)
        # pdb.set_trace()
        prob += pulp.lpSum([v_star[s] for s in states])
        for s in range(self.tot_states_num):
            for a in range(self.tot_action_num):
                prob += pulp.lpSum((self.trans_matrix[s, a, s_prime] *
                                    (self.reward_matrix[s, a, s_prime] +
                                     self.gamma * v_star[s_prime]))
                                   for s_prime in range(self.tot_states_num)) <= v_star[s], ""

        prob.writeLP('mdp2.lp')
        prob.solve()
        # print("Status:", pulp.LpStatus[prob.status])
        value_func = np.zeros(self.tot_states_num)
        for s in range(self.tot_states_num):
            value_func[s] = pulp.value(v_star[s])
            # print(pulp.value(v_star[s]))
        # print(value_func)
        policy = self.value_func_to_policy(value_func)
        str_to_print = ''
        for s in range(self.tot_states_num):
            str_to_print += self.out_format.format(value_func[s], policy[s])
        print(str_to_print)
