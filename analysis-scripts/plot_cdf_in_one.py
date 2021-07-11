import sys, os
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)


combined = np.loadtxt(sys.argv[1])

minDist = np.loadtxt(sys.argv[2])

bwAware = np.loadtxt(sys.argv[3])

routeAware = np.loadtxt(sys.argv[4])
random = np.loadtxt(sys.argv[5])
fixed = np.loadtxt(sys.argv[6])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
sns.ecdfplot(combined, label='combined')
sns.ecdfplot(minDist, label='minDist')
sns.ecdfplot(bwAware, label='bwAware')
sns.ecdfplot(routeAware, label='routeAware')
sns.ecdfplot(random, label='random')
sns.ecdfplot(fixed, label='fixed')
# plt.xlim([0.06, 0.1])
plt.xlabel("Latency (s)")
plt.ylabel("CDF")
plt.legend()
plt.savefig('latency-cdf-compare.png')