from __future__ import division
import re
import matplotlib.pyplot as plt

pat = re.compile(r'Pulls\s=\s(.*)Regret\s=\s(.*)')

tdir = './eval/epsilon-greedy/10/'

all_pulls_vs_regret = list()
for i in range(100):
    fname = tdir + 'rs_' + str(i) + '.txt'
    f = open(fname, 'rb')
    lines = f.read
    pull_vs_regret = pat.findall(lines)
    all_pulls_vs_regret.append(pull_vs_regret)
    f.close()


avg_r = list()
for pr in all_pulls_vs_regret:
    r1 = 0
    for p, r in pr:
        r1 += r
    p += 1
    avg_r.append(r1/p)
