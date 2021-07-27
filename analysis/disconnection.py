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

def plot_disconnect(disconnect_trace, run_time, num_nodes, save_dir):
    # if len(sys.argv) < 2:
    #     print("Usage: python3 analysis-scripts/plot_disconnect.py <disconnect_trace>")
    disconnect = np.loadtxt(disconnect_trace)
    if disconnect.shape[0] == 0:
        return
    elif disconnect.ndim == 1:
        disconnect = disconnect.reshape(-1, 1)
        disconnect_ids = np.array([disconnect[0]],dtype=int)
    else:
        disconnect_ids = np.array(disconnect[0],dtype=int)

    fig = plt.figure(figsize=(18,12))
    axes = []
    # cnt = 0
    for i in range(num_nodes):
        axes.append(fig.add_subplot(6, 1, i+1))
        connect = np.ones((run_time,))
        if i in disconnect_ids:
            idx = np.argwhere(disconnect_ids == i)[0][0]
            disconnect_ts = disconnect[1, idx]
            connect[int(disconnect_ts)+1:] = 0
            # cnt += 1
        axes[-1].plot(np.arange(0, len(connect)), connect, c=colors[i], label='node%d'%i)
        axes[-1].legend()
        axes[-1].set_ylim([-0.1,1.1])
        
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time (s)")
    plt.ylabel("Connect State (1 for conn, 0 for disconn)")
    plt.savefig(save_dir + 'connect-state.png')