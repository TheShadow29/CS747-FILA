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


def plot_algo(algo, lamb=0, trace='accum', tdir='../results/'):
    # tdir = '../results/' + algo + '/100000/'
    # tdir = '../results/backup/'
    # tdir = '../results/'
    if algo == 'qlearn':
        tdir += 'qlearn_rs'
    elif algo == 'sarsa':
        if trace == 'accum':
            tdir += 'sarsa_accum_lambda' + str(lamb) + '_rs'
        else:
            tdir += 'sarsa_repl_lambda' + str(lamb) + '_rs'

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

def plot_sarsa2(tdir='../results/'):
    num_files = 50
    lamb_list = [0, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9]

    for lamb in lamb_list:
        top_dir = tdir + 'sarsa_accum_lambda' + str(lamb) + '_rs'
        all_reward_list = list()
        for i in range(num_files):
            # fname = tdir + 'rs_' + str(i) + '.txt'
            reward_list = list()            #
            fname = top_dir + str(i) + '.txt'
            f = open(fname, 'r')
            lines = f.read()
            f.close()
            rlist_tmp = pat.findall(lines)
            for i in rlist_tmp:
                reward_list.append(i[0])
            all_reward_list.append(reward_list)


x1, r1 = plot_algo('qlearn', tdir='../results/instance0/')
x8, r8 = plot_algo('qlearn', tdir='../results/instance1/')
x2, r2 = plot_algo('sarsa', lamb=0, tdir='../results/instance0/')  #
x9, r9 = plot_algo('sarsa', lamb=0, tdir='../results/instance1/')
# x3, r3 = plot_algo('sarsa', 0.2)
# x4, r4 = plot_algo('sarsa', 0.4)
# x5, r5 = plot_algo('sarsa', 0.6)
# x6, r6 = plot_algo('sarsa', 0.8)
# x7, r7 = plot_algo('sarsa', 0, 'replace')
plt.plot(x1, r1, '-r', label="qlearning_inst0")
plt.plot(x8, r8, '-g', label="qlearning_inst1")
plt.plot(x2, r2, '-b', label="sarsa0_inst0")
plt.plot(x9, r9, '-y', label="sarsa0_inst1")
# plt.plot(x3, r3, '-b', label="sarsa0.2")
# plt.plot(x4, r4, '-y', label="sarsa0.4")
# plt.plot(x5, r5, '-c', label="sarsa0.6")
# plt.plot(x6, r6, '-m', label="sarsa0.8")
# plt.plot(x7, r7, '-k', label='sarsa0repl')

plt.legend(bbox_to_anchor=(0.75, 0.4), loc=2, borderaxespad=0.)
# plt.show()
# plt.savefig('Instance-25.png')
# plt.savefig('Instance-5_linear.png')
# plt.savefig('qlearn_instance_0_1')
# plt.savefig('qlearn_sarsa0_instance_0_1')
