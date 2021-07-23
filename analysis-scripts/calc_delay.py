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
receiver_throughput = {}
for i in range(num_nodes):
    receiver_ts_dict[i] = {}
    receiver_throughput[i] = []
delay_dict = {}
delay_dict_ts = {}
v2v_delay_dict = {}

def get_sender_ts(filename):
    sender_ts = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("[V2I"):
                parse = line.split()
                ts = float(parse[-1])
                frame = int(parse[-2])
                sender_ts[frame] = ts
            elif line.startswith("[V2V"):
                parse = line.split()
                ts = float(parse[-1])
                frame = int(parse[-5])
                sender_ts[frame] = ts
    return sender_ts


def get_receiver_ts(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("[Full frame recved]"):
                parse = line.split()
                sender_id = int(parse[4][:-1])
                frame = int(parse[6])
                thrpt = float(parse[8])
                ts = float(parse[-1])
                # receiver_ts_dict[sender_id].append(ts)
                receiver_throughput[sender_id].append([ts, thrpt])
                receiver_ts_dict[sender_id][frame] = ts
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

def calculate_latency(sender_ts_dict, receiver_ts_dict):
    delay_dict, delay_dict_ts = {}, {}
    for k, v in receiver_ts_dict.items():
        delay_dict[k] = v - sender_ts_dict[k]
        delay_dict_ts[sender_ts_dict[k]] = v - sender_ts_dict[k]
    return delay_dict, delay_dict_ts
    

def construct_ts_latency_array(delay_dict_ts):
    ts, delay = [], []
    for i in sorted(delay_dict_ts.keys()):
        ts.append(i)
        delay.append(delay_dict_ts[i])
    ts = np.array(ts) - np.min(ts)
    delay = np.array(delay)
    return ts, delay


def main():
    for i in range(num_nodes):
        sender_ts_dict[i] = get_sender_ts(dir + 'logs/node%d.log'%i)
        helper_ts_dict[i] = get_helper_receive_ts(dir + 'logs/node%d.log'%i)
    get_receiver_ts(dir + 'logs/server.log')
    delay_all = np.empty((frames,))
    for i in range(num_nodes):
        delay_dict[i], delay_dict_ts[i] = calculate_latency(sender_ts_dict[i], receiver_ts_dict[i])
        # print(len(delay_dict[i]))
        if i == 0:
            delay_all = np.fromiter(delay_dict[i].values(), dtype=float)
            # print(delay_all)
        else:
            delay = np.fromiter(delay_dict[i].values(), dtype=float)
            delay_all = np.concatenate((delay_all, delay))


        
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    for i in range(num_nodes):
        sns.ecdfplot(np.fromiter(delay_dict[i].values(), dtype=float), label='node%d'%i)
    # plt.xlim([0, 0.5])
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig(dir+'latency-cdf.png')

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    for i in range(num_nodes):
        # plot time series data
        ts, delay = construct_ts_latency_array(delay_dict_ts[i])
        ax.plot(ts, delay, '--o', label='node%d'%i)
        np.savetxt(dir+'node%d_delay.txt'%i, np.fromiter(delay_dict[i].values(), dtype=float))
        np.savetxt(dir+'node%d_thrpt.txt'%i, np.array(receiver_throughput[i]))

    plt.xlabel("Time (s)")
    plt.ylabel("Latency (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir+'latency-frame.png')

    # fig = plt.figure(figsize=(18,12))
    # for i in range(num_nodes):
    #     ax = fig.add_subplot(num_nodes, 1, i+1)
    #     ax.plot(np.array(receiver_throughput[i])[:,0] -np.array(receiver_throughput[i])[0][0], np.array(receiver_throughput[i])[:,1], '--o', label='node%d'%i)
    #     ax.legend()
    #     ax.set_ylim(0, 220)
    # fig.add_subplot(111, frameon=False)
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.ylabel('Throughput (Mbps)')
    # plt.xlabel('Time (s)')
    # plt.tight_layout()
    # plt.savefig(dir+'thrpt.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    sns.ecdfplot(delay_all)
    # plt.xlim([0, 0.5])
    np.savetxt(dir+'all_delay.txt', delay_all)
    # print(len(delay_all[delay_all <= 0]))
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    # plt.legend()
    plt.savefig(dir+'latency-cdf-all.png')


if __name__ == '__main__':
    main()
