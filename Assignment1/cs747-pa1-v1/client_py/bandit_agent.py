from __future__ import print_function
import argparse
import socket
from socket import error as SocketError
import errno
from bandit_algos import epsilon_greedy, ucb, kl_ucb, thompson_sampling
# import numpy as np
import pdb
import random

parser = argparse.ArgumentParser()
parser.add_argument("--numArms", type=int, help="number of arms")
parser.add_argument('--randomSeed', type=int, help='random seed')
parser.add_argument('--horizon', type=int, help='horizon number')
parser.add_argument('--hostname', type=str, help='hostname')
parser.add_argument('--port', type=int, help='port')
parser.add_argument('--algorithm', type=str, help='which algo')
parser.add_argument('--epsilon', type=float, help='epsilon for e-greedy')

args = parser.parse_args()

num_arms = 5
rand_seed = 0
horizon = 25
hostname = 'localhost'
port = 5000
algo = 'rr'
epsilon = 0.1
if args.numArms:
    num_arms = args.numArms
if args.randomSeed:
    rand_seed = args.randomSeed
if args.horizon:
    horizon = args.horizon
if args.hostname:
    hostname = args.hostname
if args.port:
    port = args.port
if args.algorithm:
    algo = args.algorithm
if args.epsilon:
    epsilon = args.epsilon

random.seed(rand_seed)


def print_reward_list(reward_list, pull_list):
    s = ''
    for ind, r in enumerate(reward_list):
        s += ' Arm: ' + str(ind) + ' : ' + str(r) + '/' + str(pull_list[ind]) + ' '
    print(s)


def sample_arm(algo, epsilon, pulls, reward, num_arms, pull_list, reward_list):
    if algo == 'rr':
        return pulls % num_arms
    elif algo == 'epsilon-greedy':
        print('Starting epsilon greedy')
        # random.seed(rand_seed)
        return epsilon_greedy(epsilon, num_arms, rand_seed, pull_list, reward_list)
    elif algo == 'UCB':
        print('Starting UCB')
        return ucb(pulls, num_arms, reward_list, pull_list)
    elif algo == 'KL-UCB':
        print('Starting KL-UCB')
        return kl_ucb(pulls, num_arms, reward_list, pull_list)
    elif algo == 'Thompson-Sampling':
        return thompson_sampling(reward_list, pull_list)
    else:
        return -1


print('Starting the code :', 'numArms', num_arms,
      'randomSeed', rand_seed, 'horizon', horizon,
      'hostname', hostname, 'port', port, 'algo', algo, 'epsilon', epsilon)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostbyname(hostname)
# guess = True
s.connect((host, port))
# while guess:
# try:
#
#     guess = False
# except Exception as e:
#     print('server socket not open yet')
#     guess = True

reward = 0
pulls = 0

reward_list = [0]*num_arms
pull_list = [0]*num_arms

arm_to_pull = sample_arm(algo, epsilon, pulls, reward, num_arms, pull_list, reward_list)

print('Sending Action', arm_to_pull)

dat = str(arm_to_pull)
# while(s.send(dat, flags=socket.MSG_NOSIGNAL) >= 0):
while(s.send(dat) >= 0):
    try:
        recv_buff = s.recv(256)
        recv_buff = recv_buff.rstrip('\x00')
        # print(recv_buff, type(recv_buff))
        reward, pulls = recv_buff.split(',')
        reward = float(reward)
        # pdb.set_trace()
        pulls = int(pulls)
        reward_list[arm_to_pull] += reward
        pull_list[arm_to_pull] += 1

        print('Received reward', reward)
        print('No. of pulls', pulls)
        print_reward_list(reward_list, pull_list)
        arm_to_pull = sample_arm(algo, epsilon, pulls, reward, num_arms, pull_list, reward_list)

        dat = str(arm_to_pull)
    # except SocketError as e:
    #     if e.errno != errno.ECONNRESET:
    #         # pdb.set_trace()
    #         raise
    #     else:
    #         # pass
    #         break
    except Exception as e:
        break


s.close()
print("Terminating")
