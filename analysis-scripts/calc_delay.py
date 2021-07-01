import sys, os
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)


np.set_printoptions(precision=3)

sender_ts_dict = {}
receiver_ts_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
delay_dict = {}

def get_sender_ts(filename):
    sender_ts = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("[V2"):
                sender_ts.append(float(line.split()[-1]))
    return sender_ts

def get_receiver_ts(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("[Full frame recved]"):
                parse = line.split()
                sender_id = int(parse[4][:-1])
                ts = float(parse[-1])
                receiver_ts_dict[sender_id].append(ts)
    


def main():
    dir = sys.argv[1]
    for i in range(6):
        sender_ts_dict[i] = get_sender_ts(sys.argv[1] + '/logs/node%d.log'%i)
    # print(sender_ts_dict)
    get_receiver_ts(sys.argv[1] + '/logs/server.log')
    # print(len(receiver_ts_dict[0]))
    for i in range(6):
        delay_dict[i] = np.array(receiver_ts_dict[i]) \
                        - np.array(sender_ts_dict[i])
        print(delay_dict[i])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    for i in range(6):
        # ax.plot(np.arange(0, len(delay_dict[i])), delay_dict[i], '--o', label='node%d'%i)
        sns.distplot(delay_dict[i], kde_kws={'cumulative': True, "lw": 2.5}, hist=False, label='node%d'%i)
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig(sys.argv[1] + '/latency-cdf.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    for i in range(6):
        ax.plot(np.arange(0, len(delay_dict[i])), delay_dict[i], '--o', label='node%d'%i)
    plt.xlabel("Frame Number")
    plt.ylabel("Latency (s)")
    plt.legend()
    plt.savefig(sys.argv[1] + '/latency-frame.png')


if __name__ == '__main__':    
    main()
