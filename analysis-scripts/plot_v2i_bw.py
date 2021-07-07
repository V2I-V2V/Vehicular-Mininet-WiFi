import sys
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import random
font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)
colors = ['r', 'b', 'maroon', 'darkblue', 'g']
if len(sys.argv) < 3:
    print("Usage: python3 analysis-scripts/plot_v2i_thrpt.py <thrpt_trace> <num_helpees>")
v2i_thrpts = np.loadtxt(sys.argv[1])
num_helpees = int(sys.argv[2])
fig = plt.figure(figsize=(12,9))
# ax = fig.add_subplot(221)
for i in range(num_helpees, v2i_thrpts.shape[1]):
    ax = fig.add_subplot(2, 2, i-num_helpees+1)
    thrpt_i = v2i_thrpts[:, i]
    # if i < num_helpees:
    #     thrpt_i[10:] = 0
    ax.plot(np.arange(0, len(thrpt_i)), thrpt_i, c=colors[i-num_helpees], label='node%d'%i)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bandwidth (Mbps)")
    ax.legend()
# plt.xlabel("Time (s)")
# plt.ylabel("Throughput (Mbps)")
# plt.legend()
plt.savefig('v2i-bw.png')