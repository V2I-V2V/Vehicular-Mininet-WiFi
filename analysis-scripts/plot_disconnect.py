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
if len(sys.argv) < 2:
    print("Usage: python3 analysis-scripts/plot_disconnect.py <disconnect_trace>")
disconnect = np.loadtxt(sys.argv[1])
disconnect_ids = np.array(disconnect[0],dtype=int)
fig = plt.figure(figsize=(18,12))
axes = []
cnt = 0
for i in range(6):
    axes.append(fig.add_subplot(6, 1, i+1))
    connect = np.ones((70,))
    if i in disconnect_ids:
        disconnect_ts = disconnect[1, cnt]
        connect[int(disconnect_ts)+1:] = 0
        cnt += 1
    axes[-1].plot(np.arange(0, len(connect)), connect, c=colors[i], label='node%d'%i)
    axes[-1].legend()
    axes[-1].set_ylim([-0.1,1.1])
    
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Time (s)")
plt.ylabel("Connect State (1 for conn, 0 for disconn)")
plt.savefig('connect-state.png')