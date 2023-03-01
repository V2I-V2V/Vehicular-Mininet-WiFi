import matplotlib
import numpy as np
matplotlib.use('AGG')
import matplotlib.pyplot as plt
plt.rc('font', family='sans-serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})
import json
from util import *

f=open('live.json')
schedule_data = json.load(f)

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
for schedule, value in schedule_data.items():
    # if schedule is not 'ccombined-adapt':
    latency, accuracy = value[0], value[1]
    ax.scatter(value[0], value[1],
        label=shced_to_displayed_name[schedule], marker=sched_to_marker[schedule],\
        color=sched_to_color[schedule])
    ax.errorbar(value[0], value[1],
                xerr=value[2], yerr=value[3], capsize=2, 
                color=sched_to_color[schedule])
    # annotate
    if shced_to_displayed_name[schedule] == 'Harbor':
        ax.annotate(shced_to_displayed_name[schedule], (latency + 0.06, accuracy-0.01), color=sched_to_color[schedule])
    elif shced_to_displayed_name[schedule] == 'no-deadline-aware':
        ax.annotate(shced_to_displayed_name[schedule], (latency + 0.065, accuracy+0.001), color=sched_to_color[schedule])
    elif shced_to_displayed_name[schedule] == 'no-prioritization':
        ax.annotate(shced_to_displayed_name[schedule], (latency -0.003, accuracy-0.003), color=sched_to_color[schedule])
    elif shced_to_displayed_name[schedule] == 'no-ddl-aware+no-prio':
        ax.annotate(shced_to_displayed_name[schedule], (latency + 0.075, accuracy-0.003), color=sched_to_color[schedule])  
    else:
        ax.annotate(shced_to_displayed_name[schedule], (latency - 0.01, accuracy+0.005), color=sched_to_color[schedule])
ax.annotate("Better", fontsize=30, horizontalalignment="center", xy=(0.375, 0.675), xycoords='data',
        xytext=(0.4, 0.65), textcoords='data',
        arrowprops=dict(arrowstyle="->, head_width=0.3", connectionstyle="arc3", lw=3)
        )
ax.grid(linestyle='--')
plt.xlabel("Detection Latency (s)")
plt.ylabel("Detection Accuracy")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig('live.pdf')

