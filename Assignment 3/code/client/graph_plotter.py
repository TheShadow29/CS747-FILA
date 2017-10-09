from __future__ import division
import re
import matplotlib.pyplot as plt
import numpy as np
import pickle

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


def plot_algo(algo):
    # tdir = '../results/' + algo + '/100000/'
    if algo == 'qlearn':
        tdir = '../results/' + algo + '/' + algo + '_rs'
    elif algo == 'sarsa':
        tdir = '../results/sarsa/sarsa_accum_lambda0_rs'
    all_reward_list = list()
    num_files = 5
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

    for pr in all_reward_list:
        r1 = r1 + np.array(pr).astype(float)

    r1 = r1 / num_files

    x_axis = np.arange(1, horizon + 1)
    return x_axis, r1
    # plt.plot(x_axis, r1)
    # plt.show()


x1, r1 = plot_algo('qlearn')
x2, r2 = plot_algo('sarsa')
# x3, r3 = plot_algo('KL-UCB')
# x4, r4 = plot_algo('Thompson-Sampling')

# with open('ins2_eps.pkl', 'w') as f:
#     pickle.dump(r1, f)
# with open('ins2_ucb.pkl', 'w') as f:
#     pickle.dump(r2, f)
# with open('ins2_klucb.pkl', 'w') as f:
#     pickle.dump(r3, f)
# with open('ins2_ths.pkl', 'w') as f:
#     pickle.dump(r4, f)

# plt.plot(np.log10(x1), r1, '-r', np.log10(x2), r2, '-g')
# with open('./eval/ins1_eps.pkl') as f:
#     r1 = pickle.load(f)
# with open('./eval/ins1_ucb.pkl') as f:
#     r2 = pickle.load(f)
# with open('./eval/ins1_klucb.pkl') as f:
#     r3 = pickle.load(f)
# with open('./eval/ins1_ths.pkl') as f:
#     r4 = pickle.load(f)
# with open('./eval/ins2_eps.pkl') as f:
#     r1 = pickle.load(f)
# with open('./eval/ins2_ucb.pkl') as f:
#     r2 = pickle.load(f)
# with open('./eval/ins2_klucb.pkl') as f:
#     r3 = pickle.load(f)
# with open('./eval/ins2_ths.pkl') as f:
#     r4 = pickle.load(f)

# plt.plot(x1, r1, '-r', x2, r2, '-g', x3, r3, '-b', x4, r4, '-y')
# x1 = np.arange(1, len(r1) + 1)
# x2 = np.log10(x1)
plt.plot(x1, r1, '-r', label="qlearning")
plt.plot(x2, r2, '-g', label="sarsa")
# plt.show()
# plt.semilogx(x1, r1, '-r', label="Epsilon-greedy")
# plt.plot(x1, r2, '-g', label='UCB')  #
# plt.semilogx(x1, r2, '-g', label='UCB')  #
# # plt.plot(x1, r3, '-b', label='KL-UCB')
# plt.semilogx(x1, r3, '-b', label='KL-UCB')
# # plt.plot(x1, r4, '-y', label='Thompson Sampling')
# plt.semilogx(x1, r4, '-y', label='Thompson Sampling')
plt.legend(bbox_to_anchor=(0.05, 1), loc=2, borderaxespad=0.)
plt.show()
# plt.savefig('Instance-25.png')
# plt.savefig('Instance-5_linear.png')
