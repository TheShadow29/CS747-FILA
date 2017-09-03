import numpy as np
from mdp_algos import mdp_solver
import random

def read_data(fname):
    # fname = '../data/MDP10.txt'
    mdp_file = open(fname, 'r')

    mdp_file_lines = mdp_file.readlines()
    mdp_file.close()

    tot_states_num = int(mdp_file_lines[0])
    tot_action_num = int(mdp_file_lines[1])

    reward_string = [m.split('\t')[:-1] for m in mdp_file_lines[2:2+tot_action_num*tot_states_num]]
    reward_fn = np.array(reward_string, dtype=np.float32).reshape((tot_states_num,
                                                                   tot_action_num, tot_states_num))
    trans_string = [m.split('\t')[:-1] for m in mdp_file_lines[2+tot_action_num*tot_states_num:-1]]
    trans_fn = np.array(trans_string, dtype=np.float32).reshape((tot_states_num,
                                                                 tot_action_num, tot_states_num))
    gamma = float(mdp_file_lines[-1])
    return tot_states_num, tot_action_num, reward_fn, trans_fn, gamma


if __name__ == '__main__':
    fname = '../data/MDP10.txt'
    tot_states_num, tot_action_num, reward_matrix, trans_matrix, gamma = read_data(fname)

    # algo = 'lp'
    algo = 'bspi'
    random_seed = 0
    batch_size = 3
    solver = mdp_solver(tot_states_num, tot_action_num, reward_matrix,
                        trans_matrix, gamma, batch_size, random_seed)
    random.seed(random_seed)
    if algo == 'lp':
        print('Linear programming')
        opt_value_fn, opt_policy = solver.linear_programming()
    elif algo == 'hpi':
        print('Howard Policy Iteration')
        opt_value_fn, opt_policy = solver.howard_pi()
    elif algo == 'rpi':
        print('Random Policy Iteration')
        opt_value_fn, opt_policy = solver.random_pi()
    elif algo == 'bspi':
        print('BSPI')
        solver.batch_switch_pi()
