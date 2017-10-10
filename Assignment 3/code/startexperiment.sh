# demonstrates how to call the server and the client
# modify according to your needs

mkdir -p results
# mkdir -p results/random
# mkdir -p results/qlearn
# mkdir -p results/sarsa

# for((n=0;n<50;n++))
# do
#	echo "----------------    Q Learning $n    ------------------"
#	# python3 ./server/server.py -port $((4000+$n)) -i 0 -rs $n -ne 1600 -q | tee "results/random_rs$n.txt" &
#	# python3 ./server/server.py -port $((4000+$n)) -i 0 -rs $n -ne 1600 -q --side 3 | tee "results/random/random_rs$n.txt" &
#	python3 ./server/server.py -port $((4000+$n)) -i 1 -rs $n -ne 1600 -q --side 32 --maxlength 1000| tee "results/qlearn_rs$n.txt" &
#	sleep 1
#	# python3 ./client/client.py -port $((4000+$n)) -rs $n -gamma 1 -algo random
#	python3 ./client/client.py -port $((4000+$n)) -rs $n -gamma 1 -algo qlearning
# done
# for((n=0;n<50;n++))
# do
#	echo "----------------    SARSA \0 $n    ------------------"
#	python3 ./server/server.py -port $((5000+$n)) -i 1 -rs $n -ne 1600 -q | tee "results/sarsa_accum_lambda0_rs$n.txt" &
#	sleep 1
#	python3 ./client/client.py -port $((5000+$n)) -rs $n -gamma 1 -algo sarsa -lambda 0
# done

# for((n=0;n<50;n++))
# do
#	echo "----------------    SARSA \0 $n    ------------------"
#	python3 ./server/server.py -port $((5000+$n)) -i 1 -rs $n -ne 1600 -q | tee "results/sarsa_repl_lambda0_rs$n.txt" &
#	sleep 1
#	python3 ./client/client.py -port $((5000+$n)) -rs $n -gamma 1 -algo sarsa -lambda 0 -trace replace
# done
# for((n=0;n<50;n++))
# do

#	lamb=0.2
#	echo "----------------    SARSA \$lamb $n    ------------------"
#	python3 ./server/server.py -port $((6000+$n)) -i 1 -rs $n -ne 1600 -q | tee "results/sarsa_accum_lambda""$lamb""_rs$n.txt" &
#	sleep 1
#	python3 ./client/client.py -port $((6000+$n)) -rs $n -gamma 1 -algo sarsa -lambda $lamb
# done

# for((n=0;n<50;n++))
# do
#	lamb=0.4

#	python3 ./server/server.py -port $((7000+$n)) -i 1 -rs $n -ne 1600 -q | tee "results/sarsa_accum_lambda""$lamb""_rs$n.txt" &
#	sleep 1
#	python3 ./client/client.py -port $((7000+$n)) -rs $n -gamma 1 -algo sarsa -lambda $lamb
# done


# for((n=0;n<50;n++))
# do
#	lamb=0.6
#	echo "----------------    SARSA \$lamb $n    ------------------"
#	python3 ./server/server.py -port $((8000+$n)) -i 1 -rs $n -ne 1600 -q | tee "results/sarsa_accum_lambda""$lamb""_rs$n.txt" &
#	sleep 1
#	python3 ./client/client.py -port $((8000+$n)) -rs $n -gamma 1 -algo sarsa -lambda $lamb
# done

# for((n=0;n<50;n++))
# do
#	lamb=0.8
#	echo "----------------    SARSA \$lamb $n    ------------------"
#	python3 ./server/server.py -port $((9000+$n)) -i 1 -rs $n -ne 1600 -q | tee "results/sarsa_accum_lambda""$lamb""_rs$n.txt" &
#	sleep 1
#	python3 ./client/client.py -port $((9000+$n)) -rs $n -gamma 1 -algo sarsa -lambda $lamb
# done
for((n=0;n<50;n++))
do
	lamb=0.05
	echo "----------------    SARSA \$lamb $n    ------------------"
	python3 ./server/server.py -port $((9000+$n)) -i 0 -rs $n -ne 1600 -q | tee "results/sarsa_accum_lambda""$lamb""_rs$n.txt" &
	sleep 1
	python3 ./client/client.py -port $((9000+$n)) -rs $n -gamma 1 -algo sarsa -lambda $lamb
done

for((n=0;n<50;n++))
do
	lamb=0.1
	echo "----------------    SARSA \$lamb $n    ------------------"
	python3 ./server/server.py -port $((4000+$n)) -i 0 -rs $n -ne 1600 -q | tee "results/sarsa_accum_lambda""$lamb""_rs$n.txt" &
	sleep 1
	python3 ./client/client.py -port $((4000+$n)) -rs $n -gamma 1 -algo sarsa -lambda $lamb
done

for((n=0;n<50;n++))
do
	lamb=0.25
	echo "----------------    SARSA \$lamb $n    ------------------"
	python3 ./server/server.py -port $((5000+$n)) -i 0 -rs $n -ne 1600 -q | tee "results/sarsa_accum_lambda""$lamb""_rs$n.txt" &
	sleep 1
	python3 ./client/client.py -port $((5000+$n)) -rs $n -gamma 1 -algo sarsa -lambda $lamb
done

for((n=0;n<50;n++))
do
	lamb=0.75
	echo "----------------    SARSA \$lamb $n    ------------------"
	python3 ./server/server.py -port $((6000+$n)) -i 0 -rs $n -ne 1600 -q | tee "results/sarsa_accum_lambda""$lamb""_rs$n.txt" &
	sleep 1
	python3 ./client/client.py -port $((6000+$n)) -rs $n -gamma 1 -algo sarsa -lambda $lamb
done

for((n=0;n<50;n++))
do
	lamb=0.9
	echo "----------------    SARSA \$lamb $n    ------------------"
	python3 ./server/server.py -port $((7000+$n)) -i 0 -rs $n -ne 1600 -q | tee "results/sarsa_accum_lambda""$lamb""_rs$n.txt" &
	sleep 1
	python3 ./client/client.py -port $((7000+$n)) -rs $n -gamma 1 -algo sarsa -lambda $lamb
done
