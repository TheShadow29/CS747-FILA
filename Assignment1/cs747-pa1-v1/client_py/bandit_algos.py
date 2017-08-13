from __future__ import division
import numpy as np
import random
import pdb
# from scipy.stats import beta
# from scipy.stats import entropy


def get_emp_means(reward_list, pull_list):
    emp_means = [0]*len(reward_list)
    for ind, p in enumerate(pull_list):
        if p != 0:
            emp_means[ind] = reward_list[ind]/p
    return emp_means


def KL(a, b):
    '''
    The formula has been taken from the paper
    d(p,q) = p * log(p/q) + (1-p)*log((1-p)/(1-q))
    '''
    # pdb.set_trace()
    if a == 0 and b == 0:
        return 0
    elif a == 0 and b == 1:
        return np.inf
    elif a == 1 and b == 0:
        return np.inf
    elif a == 1 and b == 1:
        return 0
    elif a == 0:
        return np.log(1/(1-b))
    elif a == 1:
        return np.log(1/b)
    elif b == 0 or b == 1:
        return np.inf
    else:
        # print(a,b)
        d = a * np.log(a/b) + (1 - a)*np.log((1-a)/(1-b))
        return d
    # a = np.asarray(a, dtype=np.float)
    # b = np.asarray(b, dtype=np.float)
    # return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def get_q(rhs, p_hat, u_a):
    if p_hat == 1:
        return 1
    rhs1 = rhs / u_a
    q_list = np.arange(p_hat, 1, 0.01)
    lhs_list = list()
    for q1 in q_list:
        lhs_list.append(KL(p_hat, q1))

    # pdb.set_trace()
    lhs_nlist = np.array(lhs_list)
    diff_list = lhs_nlist - rhs1
    diff_list[diff_list <= 0] = np.inf
    # pdb.set_trace()
    ind = diff_list.argmin()
    return q_list[ind]


def get_q_bin(rhs, p_hat, u_a):
    if p_hat == 1:
        return 1
    rhs1 = rhs / u_a
    thresh = 1e-6
    low = p_hat
    high = 1
    # i = 0
    while True:
        # print('Iter', i)
        # i += 1
        mid = (low + high)/2
        qkl = KL(p_hat, mid)
        diff = qkl - rhs1
        if diff > 0 and diff < thresh:
            return mid
        else:
            if diff > 0:
                high = mid
                # mid = (mid + low)/2
            else:
                low = mid
                # mid = (mid + high)/2


def draw_from_beta_dist(a, b):
    beta_dist = beta(a, b)
    return beta_dist.rvs()


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
    # random.seed(random_seed)
    emp_means = np.array(get_emp_means(reward_list, pull_list))
    rand_numb = random.random()
    print(rand_numb)
    if rand_numb > epsilon:
        print('exploit')
        # arm_x = emp_means.argmax()
        bool1 = True
        for ind, pl in enumerate(pull_list):
            if pl == 0:
                arm_x = ind
                bool1 = False
        if bool1:
            arm_x = emp_means.argmax()

    else:
        print('explore')
        arm_x = random.randint(0, len(reward_list)-1)

    print('Arm', arm_x)
    return arm_x


def ucb(pulls, num_arms, reward_list, pull_list):
    # Need to experiment with c * add_term
    if pulls >= num_arms:
        emp_means = np.array(get_emp_means(reward_list, pull_list))
        add_term = 2 * np.log(pulls) * np.divide(np.ones([1, num_arms]), pull_list)
        ucb_terms = emp_means + add_term**(0.5)
        print('ucb_terms', ucb_terms)
        return ucb_terms.argmax()
    else:
        return pulls % num_arms


def kl_ucb(pulls, num_arms, reward_list, pull_list):
    if pulls >= num_arms:
        c = 3                   # Might want to play with this parameter as well
        rhs = np.log(pulls) + c * np.log(np.log(pulls))
        emp_means = np.array(get_emp_means(reward_list, pull_list))
        qa_list = list()
        for ind, ua_t in enumerate(pull_list):
            qa_list.append(get_q_bin(rhs, emp_means[ind], ua_t))
        qa_nlist = np.array(qa_list)
        return qa_nlist.argmax()
    else:
        return pulls % num_arms


def thompson_sampling(reward_list, pull_list):
    # Can assume that #success = reward_list
    # and #failures = pull_list - reward_list
    # May want to visualize the beta distribution graphs
    succ_list = np.array(reward_list)
    fail_list = np.array(pull_list) - succ_list
    alpha_list = succ_list + 1
    beta_list = fail_list + 1
    # xa_list = list()
    # for ind, alph in enumerate(alpha_list):
    #     xa_list.append(draw_from_beta_dist(alph, beta_list[ind]))
    xa_nlist = np.random.beta(alpha_list, beta_list)
    # xa_nlist = np.array(xa_list)
    return xa_nlist.argmax()
