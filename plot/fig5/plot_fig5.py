import matplotlib
import numpy as np
matplotlib.use('AGG')
import matplotlib.pyplot as plt
plt.rc('font', family='sans-serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})

min_latency = np.load('data-11260424/min_lat.npy')
max_latency = np.load('data-11260424/max_lat.npy')
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.plot(np.arange(len(min_latency)), min_latency, '-o', label='min latency')
ax.plot(np.arange(len(max_latency)), max_latency, '-x', label='max latency')
ax.fill_between(np.arange(len(max_latency)), min_latency, max_latency, color='lime', alpha=0.3)
plt.xlabel("Frame Number")
plt.ylabel("Upload Latency (s)")
plt.tight_layout()
plt.legend()
plt.grid(linestyle='--', axis='y')
plt.savefig('latency-var-each-node.pdf')