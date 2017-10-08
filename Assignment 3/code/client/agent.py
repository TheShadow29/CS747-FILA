import numpy as np


class RandomAgent:
    def __init__(self):
        self.step = 0

    def get_action(self):
        '''samples actions in a round-robin manner'''
        self.step = (self.step + 1) % 4
        return 'up down left right'.split()[self.step]

    def observe(self, newState, reward, event):
        pass


class q_learner(object):
    def __init__(self, num_states, state, gamma, randomseed):
        # self.action_list = ['down', 'right', 'up', 'left']
        self.action_list = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.action_list)
        self.state_list = np.arange(num_states)
        # May need to init using random, but keep terminal state as 0
        # self.q_table = np.random.random((num_states, self.num_actions))
        self.q_table = np.zeros((num_states, self.num_actions))
        self.lr = 0.5
        self.new_ep_start = True
        self.ep_num = 0
        self.iter_num = 0
        self.curr_state = state
        self.next_state = None
        self.gamma = gamma
        self.curr_reward = 0
        self.curr_action = None
        self.eps_greed = 1
        self.tot_iter = 1
        # self.cum_reward = 0
        return

    def observe(self, new_state, reward, event):
        if event == 'continue':
            self.iter_num += 1
            self.tot_iter += 1
        if event == 'goal':
            # print('---------------Starting NEW EPISODE-----------------')
            self.ep_num += 1
            self.new_ep_start = True
            self.iter_num = 0
        if event == 'terminated':
            # print('Starting NEW EPISODE')
            # print('---------------Starting NEW EPISODE-----------------')
            self.ep_num += 1
            self.new_ep_start = True
            self.iter_num = 0
        self.next_state = new_state
        self.curr_reward = reward
        self.update_qtable()
        return

    def update_qtable(self):
        q_curr_s_curr_a = self.q_table[self.curr_state, self.curr_action]
        # print('Curr State Action Value', self.curr_state, self.curr_action, q_curr_s_curr_a)
        q_new_s_a = self.q_table[self.next_state, :]
        q_new_s_a_max = np.max(q_new_s_a)
        q_tmp = self.gamma * q_new_s_a_max - q_curr_s_curr_a

        new_q_curr_s_curr_a = q_curr_s_curr_a + self.lr * (self.curr_reward + q_tmp)
        self.q_table[self.curr_state, self.curr_action] = new_q_curr_s_curr_a
        # qnew_curr_s_curr_a = self.q_table[self.curr_state, self.curr_action]
        # print('New State Action Value', self.curr_state, self.curr_action, qnew_curr_s_curr_a)
        self.curr_state = self.next_state
        return

    def get_action(self):
        greedy_action = np.argmax(self.q_table[self.curr_state, :])
        random_action = np.random.randint(0, self.num_actions)
        if np.random.random() < self.eps_greed/self.tot_iter:
            self.curr_action = random_action
            # print('Taking Random Action Eps Greedy')
        else:
            self.curr_action = greedy_action
        return self.action_list[self.curr_action]

# class sarsa(object):



class Agent:
    def __init__(self, numStates, state, gamma, lamb, algorithm, randomseed):
        '''
        numStates: Number of states in the MDP
        state: The current state
        gamma: Discount factor
        lamb: Lambda for SARSA agent
        '''
        if algorithm == 'random':
            self.agent = RandomAgent()
        elif algorithm == 'qlearning':
            # raise NotImplementedError('Q Learning needs to be implemented')
            self.agent = q_learner(numStates, state, gamma, randomseed)
        elif algorithm == 'sarsa':
            raise NotImplementedError('SARSA lambda needs to be implemented')

    def get_action(self):
        '''returns the action to perform'''
        return self.agent.get_action()

    def observe(self, newState, reward, event):
        '''
        event:
            'continue'   -> The episode continues
            'terminated' -> The episode was terminated prematurely
            'goal'       -> The agent successfully reached the goal state
        '''
        self.agent.observe(newState, reward, event)
