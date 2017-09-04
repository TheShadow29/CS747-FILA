import numpy as np
from mdp_algos import mdp_solver
import random
import argparse
import pdb
import os
import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument('--isdir', type=str2bool, help='to indicate if the fname is actually a dir')
    # parser.add_argument('--compall', type=str2bool, help='to indicate if we want to compare for all')
    args = parser.parse_args()

    fname = '../data/MDP10.txt'
    batch_size = 1
    random_seed = 0
    algo = 'bspi'
    isdir = False
    # compall = False
    if args.mdp:
        fname = args.mdp
    if args.batchsize:
        batch_size = args.batchsize
    if args.randomseed:
        random_seed = args.randomseed
    if args.algorithm:
        algo = args.algorithm
    if args.isdir:
        isdir = True
    # if args.compall:
        # compall = True

    if not isdir:
        tot_states_num, tot_action_num, reward_matrix, trans_matrix, gamma = read_data(fname)

        solver = mdp_solver(tot_states_num, tot_action_num, reward_matrix,
                            trans_matrix, gamma, batch_size, random_seed)

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
            # print('BSPI')
            opt_value_fn, opt_policy, nit = solver.batch_switch_pi()

        solver.output_print(opt_value_fn, opt_policy)
        if algo != 'lp':
            print(nit)
    else:
        tot_files = len(os.listdir(fname)[:])
        hpi_nit_arr = np.zeros(tot_files)
        rpi_nit_arr = list()
        bspi_nit_arr = list()
        for ind, f in enumerate(os.listdir(fname)[:]):
            tot_states_num, tot_action_num, reward_matrix, trans_matrix, gamma = read_data(fname + f)
            batches = np.arange(1, tot_states_num, 3)
            solver = mdp_solver(tot_states_num, tot_action_num, reward_matrix,
                                trans_matrix, gamma, batch_size, random_seed)
            hpi_vf, hpi_p, hpi_nit = solver.howard_pi()
            rand_seeds = np.arange(10)
            rpi_nit_ind_arr = np.zeros(rand_seeds.shape)
            for r_ind, r in enumerate(rand_seeds):
                solver.rand_seed = r
                rpi_vf, rpi_p, rpi_nit = solver.random_pi()
                rpi_nit_ind_arr[r_ind] = rpi_nit
            bspi_nit_ind_tot = np.zeros(batches.shape)
            for b_ind, b in enumerate(batches):
                solver.batch_size = b
                bspi_vf, bspi_p, bspi_nit = solver.batch_switch_pi()
                bspi_nit_ind_tot[b_ind] = bspi_nit
                assert np.allclose(hpi_p, bspi_p)
            assert np.allclose(hpi_p, rpi_p)

            hpi_nit_arr[ind] = hpi_nit
            rpi_nit_arr.append(rpi_nit_ind_arr)
            rpi_nit_arr[ind] = rpi_nit
            bspi_nit_arr[ind] = bspi_nit
            bspi_nit_arr.append(bspi_nit_ind_tot)
            print('Iter', ind)

        print(np.mean(hpi_nit_arr), np.mean(rpi_nit_arr), np.mean(bspi_nit_arr))
        # print(np.mean(hpi_nit_arr))
        # x1 = np.arange(1, tot_files + 1)
        # plt.plot(x1, hpi_nit_arr, 'r', label='Howard')
        # plt.plot(x1, rpi_nit_arr, 'g', label='Random')
        # plt.plot(x1, bspi_nit_arr, 'b', label='BSPI')
        # plt.legend(bbox_to_anchor=(0.05, 1), loc=2, borderaxespad=0.)
        # plt.show()
