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

thrpt_dict = {}


def get_throughput(filename):
    relay_thrpt = []
    timestamps = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("[relay throughput]"):
                relay_thrpt.append(float(line.split()[-2]))
                timestamps.append(float(line.split()[-1]))
    timestamps = np.array(timestamps) - timestamps[0]
    return np.array(relay_thrpt), timestamps



def main():
    for i in range(num_nodes):
        v2v_thrpt, ts = get_throughput(dir + 'logs/node%d.log'%i)
        thrpt_dict[i] = (ts, v2v_thrpt)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(num_nodes):
        ax.plot(thrpt_dict[i][0], thrpt_dict[i][i], '-o', label='node%d'%i)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (Mbps)")
    plt.legend()
    plt.savefig(dir+'throughput.png')
