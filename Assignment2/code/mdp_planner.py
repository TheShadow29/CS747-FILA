import numpy as np
from mdp_algos import mdp_solver
import random
import argparse
import pdb


def read_data(fname):
    # fname = '../data/MDP10.txt'
    mdp_file = open(fname, 'r')

    mdp_file_lines = mdp_file.readlines()
    mdp_file.close()

    tot_states_num = int(mdp_file_lines[0])
    tot_action_num = int(mdp_file_lines[1])

    reward_string = [m.split('\t')[:-1] for m in mdp_file_lines[2:2+tot_action_num*tot_states_num]]
    # pdb.set_trace()
    reward_fn = np.array(reward_string, dtype=np.float32).reshape((tot_states_num,
                                                                   tot_action_num, tot_states_num))
    trans_string = [m.split('\t')[:-1] for m in mdp_file_lines[2+tot_action_num*tot_states_num:-1]]
    trans_fn = np.array(trans_string, dtype=np.float32).reshape((tot_states_num,
                                                                 tot_action_num, tot_states_num))
    gamma = float(mdp_file_lines[-1])
    return tot_states_num, tot_action_num, reward_fn, trans_fn, gamma


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", type=str, help="file path for mdp")
    parser.add_argument('--randomseed', type=int, help='random seed')
    parser.add_argument('--algorithm', type=str, help='which algo')
    parser.add_argument('--batchsize', type=int, help='batch size for bspi')

    args = parser.parse_args()

    fname = '../data/MDP10.txt'
    batch_size = 1
    random_seed = 0
    algo = 'bspi'
    if args.mdp:
        fname = args.mdp
    if args.batchsize:
        batch_size = args.batchsize
    if args.randomseed:
        random_seed = args.randomseed
    if args.algorithm:
        algo = args.algorithm
    tot_states_num, tot_action_num, reward_matrix, trans_matrix, gamma = read_data(fname)

    solver = mdp_solver(tot_states_num, tot_action_num, reward_matrix,
                        trans_matrix, gamma, batch_size, random_seed)
    random.seed(random_seed)
    if algo == 'lp':
        # print('Linear programming')
        opt_value_fn, opt_policy = solver.linear_programming()
    elif algo == 'hpi':
        # print('Howard Policy Iteration')
        opt_value_fn, opt_policy, nit = solver.howard_pi()
    elif algo == 'rpi':
        # print('Random Policy Iteration')
        opt_value_fn, opt_policy, nit = solver.random_pi()
    elif algo == 'bspi':
        print('BSPI')
        opt_value_fn, opt_policy, nit = solver.batch_switch_pi()

    solver.output_print(opt_value_fn, opt_policy)
    if algo != 'lp':
        print(nit)
