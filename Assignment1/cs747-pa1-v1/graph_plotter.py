from __future__ import division
import re
import matplotlib.pyplot as plt
import numpy as np
import pickle

pat = re.compile(r'Pulls\s=\s(.*)Regret\s=\s(.*)')

# algo = 'epsilon-greedy'
horizon = 100000


def plot_algo(algo):
    tdir = './eval/instance_1/' + algo + '/100000/'

    all_pulls_vs_regret = list()
    for i in range(100):
        fname = tdir + 'rs_' + str(i) + '.txt'
        f = open(fname, 'rb')
        lines = f.read()
        pull_vs_regret = pat.findall(lines)
        all_pulls_vs_regret.append(np.array(pull_vs_regret))
        f.close()

    # avg_r = list()
    r1 = np.zeros([horizon, ])

    for pr in all_pulls_vs_regret:
        r1 = r1 + pr[:, 1].astype(float)

    r1 = r1 / 100

    x_axis = np.arange(1, horizon + 1)
    return np.log10(x_axis), r1
    # plt.plot(x_axis, r1)
    # plt.show()


x1, r1 = plot_algo('epsilon-greedy')
x2, r2 = plot_algo('UCB')
x3, r3 = plot_algo('KL-UCB')
x4, r4 = plot_algo('Thompson-Sampling')

with open('ins1_eps.pkl', 'w') as f:
    pickle.dump(r1, f)
with open('ins1_ucb.pkl', 'w') as f:
    pickle.dump(r2, f)
with open('ins1_klucb.pkl', 'w') as f:
    pickle.dump(r3, f)
with open('ins1_ths.pkl', 'w') as f:
    pickle.dump(r4, f)

# plt.plot(np.log10(x1), r1, '-r', np.log10(x2), r2, '-g')
plt.plot(x1, r1, '-r', x2, r2, '-g', x3, r3, '-b', x4, r4, '-y')
plt.show()
