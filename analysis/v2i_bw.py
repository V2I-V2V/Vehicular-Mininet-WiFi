import sys, os
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.disconnection import get_connection_status
font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)
colors = ['r', 'b', 'maroon', 'darkblue', 'g', 'grey', 'cyan', 'brown', 'coral', 'lightgreen', 
          'orchid', 'navy', 'forestgreen', 'salmon', 'gold', 'lime']
# if len(sys.argv) < 3:
#     print("Usage: python3 analysis-scripts/plot_v2i_thrpt.py <thrpt_trace> <time_to_take>")

def get_nodes_v2i_bw(bw_file, time, num_nodes, helpee_conf):
    v2i_thrpts = np.loadtxt(bw_file)
    used_thrpts = v2i_thrpts[:, :num_nodes]
    if used_thrpts.shape[0] < time:
        used_thrpts = np.vstack((used_thrpts, np.tile(used_thrpts[-1], time-used_thrpts.shape[0]).reshape(-1, num_nodes)))
    conn_status = get_connection_status(helpee_conf, time, num_nodes)
    for i in range(num_nodes):
        if conn_status is not None:
            if i in conn_status.keys():
                used_thrpts[:time,i%used_thrpts.shape[1]] *= conn_status[i]
    return used_thrpts

def plot_v2i_bw(bw_file, time, num_nodes, save_dir, helpee_conf=None):
    conn_status = {}
    if helpee_conf is not None:
        conn_status = get_connection_status(helpee_conf, time, num_nodes)
    
    v2i_thrpts = np.loadtxt(bw_file)
    # print(v2i_thrpts)
    fig = plt.figure(figsize=(11,8))
    axes = []
    for i in range(num_nodes):
        axes.append(fig.add_subplot(num_nodes, 1, i+1))
        thrpt_i = v2i_thrpts[:, i%(v2i_thrpts.shape[1])]
        if len(thrpt_i) < time:
            thrpt = np.ones((time,))
            thrpt[:len(thrpt_i)] = thrpt_i
            thrpt[len(thrpt_i):] = thrpt_i[-1]
            thrpt_i = thrpt
        else:
            thrpt_i = thrpt_i[:time]
        if conn_status is not None:
            if i in conn_status.keys():
                thrpt_i *= conn_status[i]
        axes[-1].plot(np.arange(0, len(thrpt_i)), thrpt_i, c=colors[i%len(colors)], label='node%d'%i)
        axes[-1].legend()
        axes[-1].set_xlim([0, int(time)])
        # axes[-1].set_ylim([0, 50])
        
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (Mbps)")
    # plt.legend()
    plt.savefig(save_dir+'v2i-bw.png')
    
def get_disconnect_trace(thrpt, node_id):
    start = False
    rst = [node_id]
    i = 0
    for t in thrpt:
        if t == 0 and start == False:
            start = True
            rst.append(i)
        elif t != 0 and start == True:
            start = False
            rst.append(i)
        i += 1
    print(rst)


if __name__ == '__main__':
    v2i_thrpts = np.loadtxt(sys.argv[1])
    fig = plt.figure()
    fig, axes = plt.subplots(v2i_thrpts.shape[1], 1, sharex=True, sharey=True)
    bw_data = []
    for i in range(v2i_thrpts.shape[1]):
        # axes.append(fig.add_subplot(v2i_thrpts.shape[1], 1, i+1))
        thrpt_i = v2i_thrpts[:, i]
        thrpt_i[thrpt_i < 3] = 0
        bw_data += thrpt_i[:40].tolist()
        get_disconnect_trace(thrpt_i, i)
        axes[i].plot(np.arange(0, len(thrpt_i)), thrpt_i, c=colors[i%len(colors)], label='node%d'%i)
        # axes[i].legend()
        axes[i].set_xlim([0, 70])
        # axes[i].set_ylim([0, 30])
    print(np.mean(bw_data), np.std(bw_data))    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (Mbps)")
    # plt.tight_layout()
    # plt.legend()
    plt.savefig('v2i-bw.png')