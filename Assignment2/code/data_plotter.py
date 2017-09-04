import pickle
import numpy as np
import matplotlib.pyplot as plt


class holder(object):
    def __init__(self, hpi, rpi, bspi):
        self.hpi_arr = hpi
        self.rpi_arr = rpi
        self.bspi_arr = bspi


if __name__ == '__main__':
    with open('./all_c.pkl', 'rb') as f:
        h1 = pickle.load(f)
    # print(np.mean(h1.hpi_arr))
    # for hpi in hpi_arr:
    x1 = np.arange(100)
    plt.plot(x1, h1.hpi_arr, '.r', label='Howard')
    rpi_arr = np.mean(h1.rpi_arr, axis=1)
    plt.plot(x1, rpi_arr, '.g', label='Random')
    bspi_arr = np.mean(h1.bspi_arr, axis=1)
    plt.plot(x1, bspi_arr, '.b', label='BSPI')
    # plt.xlim()
    # plt.plot(x1, )
    plt.ylim(ymin=0)
    plt.xlabel('Instances')
    plt.ylabel('Average Iterations')
    plt.legend(bbox_to_anchor=(0.05, 1), loc=2, borderaxespad=0.)
    # plt.show()
    plt.savefig('Policy Iteration Plot')
