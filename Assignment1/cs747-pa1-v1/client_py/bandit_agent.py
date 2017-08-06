from __future__ import print_function
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--numArms", type=int, help="number of arms")
parser.add_argument('--randomSeed', type=int, help='random seed for the experiment')
parser.add_argument('--horizon', type=int, help='horizon number')
parser.add_argument('--hostname', type=str, help='hostname')
parser.add_argument('--port', type=int, help='port')
parser.add_argument('--algorithm', type=str, help='which algo')
parser.add_argument('--epsilon', type=float, help='epsilon for e-greedy')

args = parser.parse_args()

# for a in args:
# print(a)
print(args.numArms)
