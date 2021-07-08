import sys, os
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15}

matplotlib.rc('font', **font)

traj = np.loadtxt(sys.argv[1])

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)

for i in range(int(traj.shape[1]/2)):
    ax.plot(traj[:,2*i], traj[:,2*i+1], label='node%d'%i)

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()

plt.savefig('traj.png')

