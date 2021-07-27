import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


t1 = np.loadtxt('0_v.txt')[:, 0]
t2 = np.loadtxt('0.txt')[:, 0]

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)

ax.plot(np.arange(len(t1)), t1-t2)

plt.ylim(-0.2, 0.7)
plt.ylabel('diff in timestamp (s)')
plt.xlabel('update count')
plt.savefig('diff.png')