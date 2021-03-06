#!/bin/bash

PWD=`pwd`

# horizon=400
horizon=29
# horizon=100000
# port=5002
port=$1
nRuns=100
hostname="localhost"
banditFile="$PWD/data/instance-25.txt"
# banditFile="$PWD/data/instance-5.txt"

algorithm="rr"					#
# algorithm="epsilon-greedy"
# algorithm="UCB"
# algorithm="KL-UCB"
# algorithm="Thompson-Sampling"

# Allowed values for algorithm parameter(case-sensitive)
# 1. epsilon-greedy
# 2. UCB
# 3. KL-UCB
# 4. Thompson-Sampling
# 5. rr

epsilon=0.1

numArms=$(wc -l $banditFile | cut -d" " -f1 | xargs)

SERVERDIR=./server
CLIENTDIR=./client_py
# CLIENTDIR=./client

# OUTPUTFILE=$PWD/serverlog1.txt
OUTPUTFILE=$PWD/serverlog2.txt
# OUTPUTFILE=$PWD/eval/$horizon/$2

randomSeed=0

pushd $SERVERDIR
cmd="./startserver.sh $numArms $horizon $port $banditFile $randomSeed $OUTPUTFILE &"
#echo $cmd
# $cmd > s1.txt
$cmd
popd

sleep 1

pushd $CLIENTDIR
cmd="./startclient.sh $numArms $horizon $hostname $port $randomSeed $algorithm $epsilon &"
#echo $cmd
# $cmd > /dev/null
# $cmd > f2.txt
$cmd > f1.txt
# $cmd
popd
