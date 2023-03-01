import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

# plt.rc('font', family='sans-serif', serif='cm10')
# plt.rc('text', usetex=True)

############### Config ####################

cmap20 = plt.cm.tab20
colorlist20 = [cmap20(i) for i in range(cmap20.N)]


def parse_results(filename):
    result_dict = defaultdict(list)
    data = np.loadtxt(filename)
    for i in range(len(data)):
        num_vehicles = int(data[i][0] + data[i][1])
        computation_time = data[i][2]
        result_dict[num_vehicles].append(computation_time)
        
    computation_times = []
    computation_time_error = []
    for k in sorted(result_dict.keys()):
        computation_times.append(np.mean(result_dict[k]))
        computation_time_error.append(np.std(result_dict[k]))
    return computation_times, computation_time_error



result_dict = defaultdict(list)

bipartite_data = np.loadtxt('bipartite.txt')
for i in range(len(bipartite_data)):
    num_vehicles = int(bipartite_data[i][0] + bipartite_data[i][1])
    computation_time = bipartite_data[i][2]
    print(num_vehicles, computation_time)
    result_dict[num_vehicles].append(computation_time)


num_vehicles = []
computation_time_bipartite = []
computation_time_bipartite_error = []
for k in sorted(result_dict.keys()):
    num_vehicles.append(k)
    computation_time_bipartite.append(np.mean(result_dict[k]))
    computation_time_bipartite_error.append(np.std(result_dict[k]))

random_time, random_time_err = parse_results('random.txt')
exaustive_time, exaustive_time_err = parse_results('exaustive.txt')

plot_id = '1'
plot_name = 'ablation-computation-complexity-40'

plt.close('all')
fig, ax = plt.subplots(figsize=(4.5, 3))
# ax.plot(num_vehicles, computation_time_bipartite, color=colorlist20[0], label='Bipartite', marker='o', markersize=4, linewidth=1.5)
ax.grid(color='gainsboro', linestyle='dashed', zorder=1)
ax.errorbar(num_vehicles, computation_time_bipartite, label='bipartite', fmt='-^', yerr=computation_time_bipartite_error, color=colorlist20[0], linewidth=1.5, capsize=2, capthick=1.5, elinewidth=1.5, markeredgewidth=1.5)
ax.errorbar(num_vehicles, random_time, label='random', fmt='-x', yerr=random_time_err, color=colorlist20[2], linewidth=1.5, capsize=2, capthick=1.5, elinewidth=1.5, markeredgewidth=1.5)
ax.errorbar(num_vehicles, exaustive_time, label='exhaustive', fmt='-o', yerr=random_time_err, color=colorlist20[4], linewidth=1.5, capsize=2, capthick=1.5, elinewidth=1.5, markeredgewidth=1.5)
ax.set_yscale('log')
ax.legend(ncol=1, loc='upper center', bbox_to_anchor=(0.38, 1.03), facecolor='#dddddd', fontsize=12)
ax.set_xlabel('Number of vehicles', fontsize=12)
ax.set_ylabel('Computation time (s)', fontsize=12)
ax.set_xticks(np.arange(4, 20, 2))
ax.hlines(0.1, 4, 20, label='threshold', linestyles='dashed', colors='red', linewidth=1.5, zorder=2)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(f'{plot_id}-{plot_name}.pdf'), format='pdf', dpi=300,
            bbox_inches='tight', pad_inches=0.07)