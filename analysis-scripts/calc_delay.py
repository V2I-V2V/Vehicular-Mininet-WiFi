import sys, os
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)

MAX_FRAMES = 80

dir=sys.argv[1]
num_nodes = int(sys.argv[2])
frames = int(sys.argv[3])

np.set_printoptions(precision=3)

sender_ts_dict = {}
helper_ts_dict = {}
receiver_ts_dict = {}
for i in range(num_nodes):
    receiver_ts_dict[i] = []
delay_dict = {}
v2v_delay_dict = {}

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
        f.close()


def get_helper_receive_ts(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        helper_receive_ts = {}
        for line in lines:
            if line.startswith("[Received a full frame/oxts]"):
                parse = line.split()
                f_id, sender_id, ts = int(parse[6]), int(parse[-2]), float(parse[-1])
                if sender_id not in helper_receive_ts.keys():
                    helper_receive_ts[sender_id] = [ts]
                else:
                    helper_receive_ts[sender_id].append(ts)
        f.close()
        return helper_receive_ts

def get_server_ass(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Assignment:"):
                parse = line.split()
                helper1 = int(parse[1][1:-1])
                helper2 = int(parse[2][1:-1])
                ts = float(parse[-1])
                # receiver_ts_dict[sender_id].append(ts)
        f.close()

def main():
    for i in range(num_nodes):
        sender_ts_dict[i] = get_sender_ts(dir + 'logs/node%d.log'%i)
        helper_ts_dict[i] = get_helper_receive_ts(dir + 'logs/node%d.log'%i)
        # print(helper_ts_dict)
    # print(sender_ts_dict)
    get_receiver_ts(dir + 'logs/server.log')
    # print(len(receiver_ts_dict[0]))
    delay_all = np.empty((frames,))
    for i in range(num_nodes):
        print(len(receiver_ts_dict[i]))
        print(len(sender_ts_dict[i]))
        if len(sender_ts_dict[i]) > len(receiver_ts_dict[i]):
            print(i)
            sender_ts_dict[i] = np.array(sender_ts_dict[i][:len(receiver_ts_dict[i])])
        elif len(sender_ts_dict[i]) < len(receiver_ts_dict[i]):
            receiver_ts_dict[i] = np.array(receiver_ts_dict[i][:len(sender_ts_dict[i])])
        delay_dict[i] = np.array(receiver_ts_dict[i][:frames]) - np.array(sender_ts_dict[i][:frames])
        print(len(delay_dict[i]))
        if i == 0:
            delay_all = delay_dict[i]
        else:
            delay_all = np.concatenate((delay_all, delay_dict[i]))
        print(delay_dict[i][delay_dict[i]<0])
        print(receiver_ts_dict[i][0])
        print(receiver_ts_dict[i][-1])
        # if len(helper_ts_dict[i].values()) > 0:
        #     for helpee, ts in helper_ts_dict[i].items():
        #         v2v_delay_dict[helpee] = np.array(ts) - np.array(sender_ts_dict[helpee][-len(ts):])
        #         print(np.array(v2v_delay_dict[helpee]))
        
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    for i in range(num_nodes):
        sns.ecdfplot(delay_dict[i], label='node%d'%i)
    plt.xlim([0, 0.5])
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig(dir+'latency-cdf.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    for i in range(num_nodes):
        ax.plot(np.arange(0, len(delay_dict[i])), delay_dict[i], '--o', label='node%d'%i)
        np.savetxt(dir+'node%d_delay.txt'%i, delay_dict[i])

    plt.xlabel("Frame Number")
    plt.ylabel("Latency (s)")
    plt.legend()
    plt.savefig(dir+'latency-frame.png')

    print(delay_all.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    sns.ecdfplot(delay_all)
    # plt.xlim([0, 0.5])
    np.savetxt(dir+'all_delay.txt', delay_all)
    print(len(delay_all[delay_all <= 0]))
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    # plt.legend()
    plt.savefig(dir+'latency-cdf-all.png')


if __name__ == '__main__':
    main()
