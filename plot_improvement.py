import sys, os
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15}

matplotlib.rc('font', **font)

rtt = [20, 60, 100, 200]
improvement = [0.9396719488462679, 1.5454545454545396, 1.9090909090909065, 4.8148148148148096, ]
improvement_err = [0.08170160135561366, 0.09090909090908283, 0.1890909090909136, 0.18111111111111788,]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(rtt, improvement)
ax.errorbar(rtt, improvement, yerr=improvement_err, capsize=2)
ax.set_xticks(rtt)
ax.set_xlabel('V2I RTT (ms)')
ax.set_ylabel('Improvement of Distributed\nover Combined Sched (%)')
plt.tight_layout()
plt.savefig('distributed_improvement.png')


noise = [0, 3.5, 8, 12, 30]
improvement = [0.00, 2.4680255795363735, 2.4746083133493215, 4.413968824940042, 12.819744204636287]
improvement_err = [0.00, 0.09090909090908283, 0.1890909090909136, 0.1890909090909136, 0.1890909090909136]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(noise, improvement)
ax.errorbar(noise, improvement, yerr=improvement_err, capsize=2)
ax.set_xticks(noise)
ax.set_xlabel('GPS Location Noise (meter)')
ax.set_ylabel('Latency Increase of Combined Sched (%)')
# ax.set_ylim(0, 7)
plt.tight_layout()
plt.savefig('loc-noise.png')
