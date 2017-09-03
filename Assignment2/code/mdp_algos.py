import numpy as np
# from pulp import *
import pulp
import pdb
import random


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
        out_format = '{0:.6f} {1:d}\n'
        str_to_print = ''
        for s in range(self.tot_states_num):
            # pdb.set_trace()
            str_to_print += out_format.format(value_func[s], policy[s])
        print(str_to_print)

    def policy_to_value_fn(self, policy):
        prob = pulp.LpProblem('mdp_lp', pulp.LpMinimize)
        states = np.arange(self.tot_states_num)
        v_pi = pulp.LpVariable.dicts('v_pi', states)
        prob += 0
        # pdb.set_trace()
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
        value_fn = self.policy_to_value_fn(policy)
        # self.output_print(value_func, policy)
        # pdb.set_trace()
        return value_fn, policy

    def q_pi(self, s, a, value_fn):
        return sum([self.trans_matrix[s, a, s_prime] *
                    (self.reward_matrix[s, a, s_prime] +
                     self.gamma * value_fn[s_prime])
                    for s_prime in range(self.tot_states_num)])

    def modify_policy(self, policy_curr, U):
        policy_new = policy_curr.copy()
        for s, a in U:
            policy_new[s] = a
        return policy_new

    def get_t_pi(self, value_fn, eps):
        t_pi = list()
        for s in range(self.tot_states_num):
            for a in range(self.tot_action_num):
                if self.q_pi(s, a, value_fn) - value_fn[s] > eps:
                    t_pi.append((s, a))

        return t_pi

    def howard_pi(self):
        eps = 1e-6
        policy_curr = np.zeros(self.tot_states_num, dtype=int)
        it = 0
        while True:
            it += 1
            print(it)
            value_fn = self.policy_to_value_fn(policy_curr)
            t_pi = self.get_t_pi(value_fn, eps)
            # print(it, len(t_pi))
            # print(it, value_fn[0])
            # pdb.set_trace()
            if len(t_pi) == 0:
                break
            else:
                policy_curr = self.modify_policy(policy_curr, t_pi)

        # self.output_print(value_fn, policy_curr)
        return value_fn, policy_curr, it

    def random_pi(self):
        eps = 1e-6
        policy_curr = np.zeros(self.tot_states_num, dtype=int)
        it = 0
        while True:
            it += 1
            value_fn = self.policy_to_value_fn(policy_curr)
            t_pi = self.get_t_pi(value_fn, eps)
            if len(t_pi) == 0:
                break
            else:
                U = list()
                for i in range(len(t_pi)):
                    if random.random() > 0.5:
                        U.append(t_pi[i])
                policy_curr = self.modify_policy(policy_curr, U)

        # self.output_print(value_fn, policy_curr)
        return value_fn, policy_curr, it

    def batch_switch_pi(self):
        b = self.batch_size
        eps = 1e-6
        policy_curr = np.zeros(self.tot_states_num, dtype=int)
        states_list = np.arange(self.tot_states_num)
        batch_list = np.split(states_list, np.arange(b, self.tot_states_num, b))
        # pdb.set_trace()
        it = 0
        while True:
            it += 1
            print(it)
            value_fn = self.policy_to_value_fn(policy_curr)
            t_pi = self.get_t_pi(value_fn, eps)
            U = [u for u, a in t_pi]
            if len(t_pi) == 0:
                break
            else:
                j = int(np.ceil(self.tot_states_num/b)) - 1
                # pdb.set_trace()
                while min(batch_list[j]) > max(U):
                    j = j-1
                u_list = list()
                for ind, i in enumerate(U):
                    if i in batch_list[j]:
                        u_list.append(t_pi[ind])
                policy_curr = self.modify_policy(policy_curr, u_list)

        # self.output_print(value_fn, policy_curr)
        return value_fn, policy_curr, it
