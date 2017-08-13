from subprocess import call
from subprocess import Popen
import os
# import argparse

# $1 : randomseed
# $2 : outfile
# $3 : horizon
# $4 : algo
# call(['./exp.sh', '0', 'eval/400/sr2.txt',])
# horizon = 10
horizon = 100000
# port = 5002
# port = 5003
# port = 5004
# algo = 'epsilon-greedy'
algo = 'UCB'
# algo = 'KL-UCB'
# algo = 'Thompson-Sampling'
if algo == 'UCB':
    port = 5002
elif algo == 'KL-UCB':
    port = 6003
elif algo == 'Thompson-Sampling':
    port = 7004
else:
    port = 8005

dir_path = './eval/instance_2/' + algo + '/' + str(horizon)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = dir_path[2:]
# fp = 'eval/epsilon-greedy/10/rs_1.txt'
# call(['./exp.sh', '0', fp, str(horizon), algo])
print(dir_path)
for i in range(2):
    # i denotes the random seed
    file_path = dir_path + '/rs_' + str(i) + '.txt'
    print(file_path)
    call(['./exp.sh', str(i), file_path, str(horizon), algo, str(port + i)])


# procs = list()
# for i in range(30, 60):
#     # i denotes the random seed
#     file_path = dir_path + '/rs_' + str(i) + '.txt'
#     print(file_path)
#     p_args = ['./exp.sh', str(i), file_path, str(horizon), algo, str(port + i)]
#     p = Popen(p_args)
#     procs.append(p)

# return_codes = [p1.wait() for p1 in procs]
