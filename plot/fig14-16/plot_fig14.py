import matplotlib
import numpy as np
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='sans-serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 27})
import json
from util import *

f=open('assignment.json')
schedule_data = json.load(f)

sched = ['combined-adapt', 'random-adapt', 'minDist-adapt', 'bwAware-adapt', 'routeAware-adapt', 'distributed-adapt']
fig = plt.figure(figsize=(6.5,5.2))
ax = fig.add_subplot(111)
for schedule in sched:
    if schedule == 'distributed-adapt' or schedule == 'routeAware-adapt':
        ls = linestyles[schedule]  
    elif schedule in sched_to_line_style:
        ls = sched_to_line_style[schedule]
    else:
        ls = '-'
    sns.ecdfplot(schedule_data[schedule], ls=ls, color=sched_to_color[schedule], label=shced_to_displayed_name[schedule])
plt.legend(loc='lower right', handlelength=0.8, handletextpad=0.1, borderpad=0.1, bbox_to_anchor=(1.02, -0.05), labelspacing=0.1)
plt.grid(linestyle='--')
handles, labels = plt.gca().get_legend_handles_labels()
print(labels)
plt.xlabel("Detection Latency (s)", fontsize=30)   
plt.ylabel("CDF", fontsize=30) 
plt.tight_layout()
plt.savefig('assignment.pdf')