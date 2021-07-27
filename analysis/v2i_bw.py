import sys
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import random
font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)
colors = ['r', 'b', 'maroon', 'darkblue', 'g', 'grey']
# if len(sys.argv) < 3:
#     print("Usage: python3 analysis-scripts/plot_v2i_thrpt.py <thrpt_trace> <time_to_take>")

def plot_v2i_bw(bw_file, time, num_nodes, save_dir):
    v2i_thrpts = np.loadtxt(bw_file)
    print(v2i_thrpts)
    fig = plt.figure(figsize=(18,12))
    axes = []
    for i in range(num_nodes):
        axes.append(fig.add_subplot(num_nodes, 1, i+1))
        thrpt_i = v2i_thrpts[:, i]
        if len(thrpt_i) < time:
            thrpt = np.ones((time,))
            thrpt[:len(thrpt_i)] = thrpt_i
            thrpt[len(thrpt_i):] = thrpt_i[-1]
            thrpt_i = thrpt
        axes[-1].plot(np.arange(0, len(thrpt_i)), thrpt_i, c=colors[i], label='node%d'%i)
        axes[-1].legend()
        axes[-1].set_xlim([0, int(time)])
        axes[-1].set_ylim([0, 220])
        
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (Mbps)")
    # plt.legend()
    plt.savefig(save_dir+'v2i-bw.png')


if __name__ == '__main__':
    v2i_thrpts = np.loadtxt(sys.argv[1])
    fig = plt.figure(figsize=(18,12))
    axes = []
    for i in range(v2i_thrpts.shape[1]):
        axes.append(fig.add_subplot(v2i_thrpts.shape[1], 1, i+1))
        thrpt_i = v2i_thrpts[:, i]
        axes[-1].plot(np.arange(0, len(thrpt_i)), thrpt_i, c=colors[i], label='node%d'%i)
        axes[-1].legend()
        axes[-1].set_xlim([0, 70])
        axes[-1].set_ylim([0, 220])
        
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (Mbps)")
    # plt.legend()
    plt.savefig('v2i-bw.png')