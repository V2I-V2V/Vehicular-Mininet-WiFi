import numpy as np
import os, sys
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import unique
import matplotlib
font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)

filename = sys.argv[1]

overhead_array = np.loadtxt(filename)
num_helpees = overhead_array[:, 0]
num_helpers = overhead_array[:, 1]
combined_computation_time = overhead_array[:, 2]
random_computation_time = overhead_array[:, 3]

unique_helpees = np.unique(num_helpees)


fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 4.2))
indices = np.argwhere(num_helpees==2)
ax.plot(np.arange(len(indices)), combined_computation_time[indices], '-o', label='combined scheduler')
# axes[cnt].axhline(0.2, linestyle='--', color='r')
ax.set_ylabel('Scheduling Computation Time (s)')
ax.legend()
plt.xlabel('Number of Helpers')
plt.tight_layout()
plt.savefig('overhead-2nodes.png')

# fig, axes = plt.subplots(len(unique_helpees), 1, sharex=True, figsize=(6, 10))

# cnt = 0
# for num_helpee in unique_helpees:
#     print(num_helpee)
#     indices = np.argwhere(num_helpees==num_helpee)
#     print(indices)
#     axes[cnt].plot(np.arange(len(indices)), combined_computation_time[indices], '-o', label='combined')
#     axes[cnt].plot(np.arange(len(indices)), random_computation_time[indices], '-o', label='random')
#     # axes[cnt].set_xticks(np.arange(len(combined_computation_time[indices])))
#     # axes[cnt].set_xticklabels(num_helpers)
#     axes[cnt].set_ylabel('%d helpees\nlatency (s)'%num_helpee)
#     axes[cnt].legend()
#     axes[cnt].axhline(0.2, linestyle='--', color='r')
#     cnt += 1


# plt.xlabel('Number of Helpers')
# plt.tight_layout()
# plt.savefig('overhead.png')