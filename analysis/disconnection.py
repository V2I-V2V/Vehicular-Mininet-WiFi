import sys
from typing_extensions import runtime
import numpy as np
import matplotlib
from numpy.lib.shape_base import tile
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import random
font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)
colors = ['r', 'b', 'maroon', 'darkblue', 'g', 'grey', 'cyan', 'brown', 'coral', 'lightgreen', 'orchid', 'navy', 'forestgreen']

def get_disconnect_id_durations(disconnect_array):
    if disconnect_array.shape[0] == 0:
        return
    elif disconnect_array.ndim == 1:
        disconnect_array = disconnect_array.reshape(-1, 1)
        disconnect_ids = np.array([disconnect_array[0]],dtype=int)
    else:
        disconnect_ids = np.array(disconnect_array[0],dtype=int)
    id_to_disconnect_tses = {}
    for idx in range(disconnect_array.shape[1]):
        id_to_disconnect_tses[disconnect_array[0][idx]] = \
            disconnect_array[1:, idx]
    return id_to_disconnect_tses


def get_disconect_duration_in_percentage(disconnect_trace, run_time, num_nodes):
    disconnect = np.loadtxt(disconnect_trace)
    id_to_disconnect_tses = get_disconnect_id_durations(disconnect)
    # sum disconnect time for nodes
    sum_disconnect_time = 0
    for node_id, disconnect_array in id_to_disconnect_tses.items():
        for i in range(0, len(disconnect_array), 2):
            if disconnect_array[i] >= run_time:
                break
            if i+1 < len(disconnect_array):
                sum_disconnect_time += min(run_time-disconnect_array[i],\
                                            disconnect_array[i+1]-disconnect_array[i])
    num_helpees = len(id_to_disconnect_tses.keys())
    # print("num helpees", num_helpees)
    # print("sum_disconnect_time", sum_disconnect_time)
    disconnect_percentage = sum_disconnect_time / (run_time * num_nodes) * 100.0
    return num_helpees, disconnect_percentage


def get_connection_status(disconnect_trace, run_time, num_nodes):
    disconnect = np.loadtxt(disconnect_trace)
    if disconnect.shape[0] == 0:
        return
    elif disconnect.ndim == 1:
        disconnect = disconnect.reshape(-1, 1)
        disconnect_ids = np.array([disconnect[0]],dtype=int)
    else:
        disconnect_ids = np.array(disconnect[0],dtype=int)
        
    
    node_to_conn_array = {}
    for i in range(num_nodes):
        connect = np.ones((run_time,))
        if i in disconnect_ids:
            idx = np.argwhere(disconnect_ids == i)[0][0]
            for cnt in range(1,disconnect.shape[0]):
                disconnect_ts = disconnect[cnt, idx]
                if cnt % 2 == 1:
                    connect[int(disconnect_ts)+1:] = 0
                else:
                    connect[int(disconnect_ts)+1:] = 1
        node_to_conn_array[i] = connect
    return node_to_conn_array

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

    fig = plt.figure(figsize=(12,8))
    axes = []
    # cnt = 0
    for i in range(num_nodes):
        axes.append(fig.add_subplot(num_nodes, 1, i+1))
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