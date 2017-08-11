from subprocess import call
import os

# $1 : randomseed
# $2 : outfile
# $3 : horizon
# $4 : algo
# call(['./exp.sh', '0', 'eval/400/sr2.txt',])
# horizon = 10
horizon = 100000
# algo = 'epsilon-greedy'
# algo = 'UCB'
# algo = 'KL-UCB'
algo = 'Thompson-Sampling'
dir_path = './eval/' + algo + '/' + str(horizon)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = dir_path[2:]
# fp = 'eval/epsilon-greedy/10/rs_1.txt'
# call(['./exp.sh', '0', fp, str(horizon), algo])
print(dir_path)
for i in range(100):
    # i denotes the random seed
    file_path = dir_path + '/rs_' + str(i) + '.txt'
    print(file_path)
    call(['./exp.sh', str(i), file_path, str(horizon), algo])
