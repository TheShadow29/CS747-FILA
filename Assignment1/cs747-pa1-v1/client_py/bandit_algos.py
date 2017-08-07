from __future__ import division
import numpy as np
import random


def get_emp_means(reward_list, pull_list):
    emp_means = [0]*len(reward_list)
    for ind, p in enumerate(pull_list):
        if p != 0:
            emp_means[ind] = reward_list[ind]/p
    return emp_means


def epsilon_greedy(epsilon, num_arms, random_seed, pull_list, reward_list):
    '''
    The algo works as follows:
    For some epsilon in [0,1]
    Choose the bandit with highest empirical mean with probability 1 - epsilon
    Choose a random bandit with probability epsilon
    Epsilon is a constant given by the user
    It returns the arm to be pulled
    '''
    # Need to experiment with different epsilon
    random.seed(random_seed)
    emp_means = np.array(get_emp_means(reward_list, pull_list))
    rand_numb = random.random()
    if rand_numb > epsilon:
        arm_x = emp_means.argmax()
    else:
        arm_x = random.randint(0, len(reward_list)-1)

    return arm_x


def ucb(pulls, num_arms, reward_list, pull_list):
    # Need to experiment with c * add_term
    if pulls >= num_arms:
        emp_means = np.array(get_emp_means(reward_list, pull_list))
        add_term = 2 * np.log(pulls) * np.divide(np.ones([1, num_arms]), pull_list)
        ucb_terms = emp_means + add_term
        return ucb_terms.argmax()
    else:
        return pulls % num_arms
