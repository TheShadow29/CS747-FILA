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
        return

    def output_print(self, value_func, policy):
        out_format = '{} {}\n'
        str_to_print = ''
        for s in range(self.tot_states_num):
            str_to_print += out_format.format(value_func[s], policy[s])
        print(str_to_print)

    def policy_to_value_fn(self, policy):
        prob = pulp.LpProblem('mdp_lp', pulp.LpMinimize)
        states = np.arange(self.tot_states_num)
        v_pi = pulp.LpVariable.dicts('v_star', states)
        prob += 0
        for s in range(self.tot_states_num):
            prob += pulp.lpSum((self.trans_matrix[s, policy[s], s_prime] *
                                (self.reward_matrix[s, policy[s], s_prime] +
                                 self.gamma * v_pi[s_prime]))
                               for s_prime in range(self.tot_states_num)) == v_pi[s], ""
        prob.writeLP('v_pi.lp')
        prob.solve()
        # print("Status:", pulp.LpStatus[prob.status])
        value_fn = np.zeros(self.tot_states_num)
        for s in range(self.tot_states_num):
            value_fn[s] = pulp.value(v_pi[s])

        return value_fn

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
        self.output_print(value_func, policy)
        return value_func, policy

    # def howard_pi(self):
    #     policy_init = np.zeros(self.tot_states_num)
