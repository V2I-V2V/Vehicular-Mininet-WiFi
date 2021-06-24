import sys
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15}

matplotlib.rc('font', **font)

positions = np.loadtxt(sys.argv[1]).reshape(-1,2)

x_loc = positions[:,0]
y_loc = positions[:,1]


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)

for i in range(len(x_loc)):
    ax.scatter(x_loc[i], y_loc[i], s=60, label='Node %d' % i)

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig('position.png')