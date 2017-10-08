# demonstrates how to call the server and the client
# modify according to your needs

mkdir -p results
mkdir -p results/random
mkdir -p results/qlearn
mkdir -p results/sarsa

# for((n=0;n<1;n++))
# do
#	echo "----------------    Q Learning $n    ------------------"
#	# python3 ./server/server.py -port $((4000+$n)) -i 0 -rs $n -ne 1600 -q | tee "results/random_rs$n.txt" &
#	# python3 ./server/server.py -port $((4000+$n)) -i 0 -rs $n -ne 1600 -q --side 3 | tee "results/random/random_rs$n.txt" &
#	python3 ./server/server.py -port $((4000+$n)) -i 0 -rs $n -ne 1600 -q --side 32 --noobfuscate --maxlength 1000| tee "results/qlearn/qlearn_rs$n.txt" &
#	sleep 1
#	# python3 ./client/client.py -port $((4000+$n)) -rs $n -gamma 1 -algo random
#	python3 ./client/client.py -port $((4000+$n)) -rs $n -gamma 1 -algo qlearning
# done
for((n=0;n<1;n++))
do
	echo "----------------    SARSA \0 $n    ------------------"
	python3 ./server/server.py -port $((5000+$n)) -i 0 -rs $n -ne 1600 -q | tee "results/sarsa/sarsa_accum_lambda0_rs$n.txt" &
	sleep 1
	python3 ./client/client.py -port $((5000+$n)) -rs $n -gamma 1 -algo sarsa -lambda 0.3
done
# for((n=0;n<10;n++))
# do
#     echo "----------------    SARSA \0.2 $n    ------------------"
#     python3 ./server/server.py -port $((6000+$n)) -i 5 -rs $n -ne 1600 -q | tee "results/sarsa_accum_lambda0.2_rs$n.txt" &
#     sleep 1
#     python3 ./client/client.py -port $((6000+$n)) -rs $n -gamma 1 -algo sarsa -lambda 0.2
# done
