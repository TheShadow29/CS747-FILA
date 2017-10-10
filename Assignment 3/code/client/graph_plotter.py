from __future__ import division
import re
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pdb
# pat = re.compile(r'Pulls\s=\s(.*)Regret\s=\s(.*)')
# pat = re.compile(r'\[(.*)\]')
pat = re.compile(r'(\d*|-\d*?)(,|\])')
# algo = 'epsilon-greedy'
# horizon = 100000

# def get_list(lines):
#     a = pat.findall(lines)
#     a1 = list()
#     for i in a:
#         a1.append(int(i[0]))
#     return a1


def plot_algo(algo, lamb=0):
    # tdir = '../results/' + algo + '/100000/'
    if algo == 'qlearn':
        tdir = '../results/qlearn_rs'
    elif algo == 'sarsa':
        tdir = '../results/sarsa_accum_lambda' + str(lamb) + '_rs'
    all_reward_list = list()
    num_files = 50
    for i in range(num_files):
        # fname = tdir + 'rs_' + str(i) + '.txt'
        reward_list = list()            #
        fname = tdir + str(i) + '.txt'
        f = open(fname, 'r')
        lines = f.read()
        f.close()
        rlist_tmp = pat.findall(lines)
        for i in rlist_tmp:
            reward_list.append(i[0])
        all_reward_list.append(reward_list)
        # reward_list = get_list(lines)
        # pull_vs_regret = pat.findall(lines)
        # all_pulls_vs_regret.append(np.array(pull_vs_regret))

    # avg_r = list()
    horizon = len(reward_list)
    r1 = np.zeros([horizon, ])
    # pdb.set_trace()
    try:
        for ind, pr in enumerate(all_reward_list):
            print(ind)
            r1 = r1 + np.array(pr).astype(float)
    except Exception as e:
        pdb.set_trace()
    r1 = r1 / num_files

    x_axis = np.arange(1, horizon + 1)
    return x_axis, r1
    # plt.plot(x_axis, r1)
    # plt.show()


x1, r1 = plot_algo('qlearn')
# x2, r2 = plot_algo('sarsa', 0)
x3, r3 = plot_algo('sarsa', 0.2)
x4, r4 = plot_algo('sarsa', 0.4)
x5, r5 = plot_algo('sarsa', 0.6)
x6, r6 = plot_algo('sarsa', 0.8)
plt.plot(x1, r1, '-r', label="qlearning")
# plt.plot(x2, r2, '-g', label="sarsa0")
plt.plot(x3, r3, '-b', label="sarsa0.2")
plt.plot(x4, r4, '-y', label="sarsa0.4")
plt.plot(x5, r5, '-c', label="sarsa0.6")
plt.plot(x6, r6, '-m', label="sarsa0.8")
plt.legend(bbox_to_anchor=(0.85, 0.4), loc=2, borderaxespad=0.)
plt.show()
# plt.savefig('Instance-25.png')
# plt.savefig('Instance-5_linear.png')
