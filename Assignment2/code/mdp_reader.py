import numpy as np

fname = '../data/MDP10.txt'
mdp_file = open(fname, 'r')

mdp_file_lines = mdp_file.readlines()
mdp_file.close()

tot_states_num = int(mdp_file_lines[0])
tot_action_num = int(mdp_file_lines[1])

reward_string = [m.split('\t')[:-1] for m in mdp_file_lines[2:2+tot_action_num*tot_states_num]]
reward_fn = np.array(reward_string, dtype=np.float32).reshape((tot_states_num,
                                                               tot_action_num, tot_states_num))
trans_string = [m.split('\t')[:-1] for m in mdp_file_lines[2+tot_action_num*tot_states_num:-1]]
trans_fn = np.array(trans_string, dtype=np.float32).reshape((tot_states_num,
                                                             tot_action_num, tot_states_num))
gamma = mdp_file_lines[-1]
# reward_fn = np.zeros((tot_states_num, tot_action_num, tot_states_num))
# trans_fn = np.zeros((tot_states_num, tot_action_num, tot_states_num))
