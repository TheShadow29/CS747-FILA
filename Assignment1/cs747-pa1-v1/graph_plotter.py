from __future__ import division
import re
import matplotlib.pyplot as plt
import numpy as np

pat = re.compile(r'Pulls\s=\s(.*)Regret\s=\s(.*)')

# algo = 'epsilon-greedy'
horizon = 100000


def plot_algo(algo):
    tdir = './eval/' + algo + '/100000/'

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
    plt.plot(x_axis, r1)
    plt.show()


plot_algo('epsilon-greedy')
