import sys
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15}

matplotlib.rc('font', **font)

if len(sys.argv) < 3:
    print("Usage: python3 analysis-scripts/plot_v2i_thrpt.py <thrpt_trace> <num_helpees>")

v2i_thrpts = np.loadtxt(sys.argv[1])
num_helpees = int(sys.argv[2])

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(v2i_thrpts.shape[1]):
    thrpt_i = v2i_thrpts[:, i]
    if i < num_helpees:
        thrpt_i[10:] = 0
    ax.plot(np.arange(0, len(thrpt_i)), thrpt_i, label='node%d'%i)
    
plt.xlabel("Time (s)")
plt.ylabel("Throughput (Mbps)")
plt.legend()
plt.savefig('v2i-bw.png')
    
