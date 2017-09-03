import numpy as np
import os


def create_mdp(fname, tot_states_num=50, tot_action_num=2):
    # tot_states_num = 2
    # tot_action_num = 2
    trans_mat = np.random.rand(tot_states_num, tot_action_num, tot_states_num)
    reward_mat = np.random.rand(tot_states_num, tot_action_num, tot_states_num)*2 - 1
    gamma = np.random.rand()
    if gamma == 1:
        gamma = 0.99

    str_format1 = '{}\n'
    str_to_write = ''
    str_to_write += str_format1.format(tot_states_num)
    str_to_write += str_format1.format(tot_action_num)

    for s in range(tot_states_num):
        for a in range(tot_action_num):
            for s_prime in range(tot_states_num):
                str_to_write += str(reward_mat[s, a, s_prime]) + '\t'
            str_to_write += '\n'
    for s in range(tot_states_num):
        for a in range(tot_action_num):
            for s_prime in range(tot_states_num):
                str_to_write += str(trans_mat[s, a, s_prime]) + '\t'
            str_to_write += '\n'
    str_to_write += str_format1.format(gamma)
    with open(fname, 'w') as f:
        f.write(str_to_write)

    return


np.random.seed(0)
tdir = './mdp_files/'
if not os.path.exists(tdir):
    os.makedirs(tdir)

for i in range(100):
    fname = tdir + str(i) + '.txt'
    create_mdp(fname, 50, 2)
