import matplotlib
import numpy as np
matplotlib.use('AGG')
import matplotlib.pyplot as plt
plt.rc('font', family='sans-serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 27})
import json
from util import *


def plot_one_category_ax0(schedule_data, ax, plot_xlabel=True, plot_ylabel=True, y_scale=1.0, left=False):
    for schedule in schedule_data.keys():
        ax.scatter(schedule_data[schedule][0], schedule_data[schedule][1],
            label='Harbor' if schedule == 'combined-adapt' else schedule, marker=sched_to_marker[schedule],\
            color=sched_to_color[schedule])
        ax.errorbar(schedule_data[schedule][0], schedule_data[schedule][1], \
            xerr=schedule_data[schedule][2], yerr=schedule_data[schedule][3], capsize=2,
            color=sched_to_color[schedule])
        
        if schedule =='v2i':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] + 0.06, schedule_data[schedule][1]+0.01*y_scale), color=sched_to_color[schedule])
        elif schedule =='v2v-adapt':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] + 0.15, schedule_data[schedule][1]+0.05*y_scale), color=sched_to_color[schedule])
        elif schedule =='v2i-adapt':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] - 0.01, schedule_data[schedule][1]+0.01*y_scale), color=sched_to_color[schedule])
        elif schedule =='v2v':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] + 0.07, schedule_data[schedule][1]-0.025*y_scale), color=sched_to_color[schedule])
        else:
            if(left):
                ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] + 0.105, schedule_data[schedule][1] + 0.004*y_scale), color=sched_to_color[schedule])
            else:
                ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] - 0.005, schedule_data[schedule][1] + 0.004*y_scale), color=sched_to_color[schedule])
    if(plot_xlabel):
        ax.set_xlabel("Detection Latency (s)", fontsize=30)
    if(plot_ylabel):
        ax.set_ylabel("Detection Accuracy", loc='top', fontsize=30)
    ax.grid(linestyle='--')

def plot_one_category_ax1(schedule_data, ax, plot_xlabel=True, plot_ylabel=True, y_scale=1.0, left=False):
    for schedule in schedule_data.keys():
        ax.scatter(schedule_data[schedule][0], schedule_data[schedule][1],
            label='Harbor' if schedule == 'combined-adapt' else schedule, marker=sched_to_marker[schedule],\
            color=sched_to_color[schedule])
        ax.errorbar(schedule_data[schedule][0], schedule_data[schedule][1], \
            xerr=schedule_data[schedule][2], yerr=schedule_data[schedule][3], capsize=2,
            color=sched_to_color[schedule])
        
        if schedule =='v2i':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] + 0.06, schedule_data[schedule][1]-0.04*y_scale), color=sched_to_color[schedule])
        elif schedule =='v2v-adapt':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] + 0.15, schedule_data[schedule][1]+0.05*y_scale), color=sched_to_color[schedule])
        elif schedule =='v2i-adapt':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] - 0.01, schedule_data[schedule][1]+0.01*y_scale), color=sched_to_color[schedule])
        elif schedule =='v2v':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] + 0.07, schedule_data[schedule][1]-0.04*y_scale), color=sched_to_color[schedule])
        else:
            if(left):
                ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] + 0.105, schedule_data[schedule][1] + 0.004*y_scale), color=sched_to_color[schedule])
            else:
                ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] - 0.005, schedule_data[schedule][1] + 0.004*y_scale), color=sched_to_color[schedule])
    if(plot_xlabel):
        ax.set_xlabel("Detection Latency (s)", fontsize=30)
    if(plot_ylabel):
        ax.set_ylabel("Detection Accuracy", loc='top', fontsize=30)
    ax.grid(linestyle='--')

def plot_one_category_ax2(schedule_data, ax, plot_xlabel=True, plot_ylabel=True, y_scale=1.0, left=False):
    for schedule in schedule_data.keys():
        ax.scatter(schedule_data[schedule][0], schedule_data[schedule][1],
            label='Harbor' if schedule == 'combined-adapt' else schedule, marker=sched_to_marker[schedule],\
            color=sched_to_color[schedule])
        ax.errorbar(schedule_data[schedule][0], schedule_data[schedule][1], \
            xerr=schedule_data[schedule][2], yerr=schedule_data[schedule][3], capsize=2,
            color=sched_to_color[schedule])
        
        if schedule =='v2i':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] + 0.06, schedule_data[schedule][1]-0.04*y_scale), color=sched_to_color[schedule])
        elif schedule =='v2v-adapt':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] - 0.01, schedule_data[schedule][1]-0.04*y_scale), color=sched_to_color[schedule])
        elif schedule =='v2i-adapt':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] - 0.01, schedule_data[schedule][1]-0.04*y_scale), color=sched_to_color[schedule])
        elif schedule =='v2v':
            ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] + 0.07, schedule_data[schedule][1]+0.025*y_scale), color=sched_to_color[schedule])
        else:
            if(left):
                ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] + 0.105, schedule_data[schedule][1] + 0.004*y_scale), color=sched_to_color[schedule])
            else:
                ax.annotate(shced_to_displayed_name[schedule], (schedule_data[schedule][0] - 0.005, schedule_data[schedule][1] + 0.004*y_scale), color=sched_to_color[schedule])
    if(plot_xlabel):
        ax.set_xlabel("Detection Latency (s)", fontsize=30)
    if(plot_ylabel):
        ax.set_ylabel("Detection Accuracy", loc='top', fontsize=30)
    ax.grid(linestyle='--')

f=open('v2i.json')
v2i_data = json.load(f)
f=open('similar.json')
similar_data = json.load(f)
f=open('v2v.json')
v2v_data = json.load(f)

fig, axes = plt.subplots(1, 3, sharex=True, figsize=(18,5))

plot_one_category_ax0(v2i_data, axes[0], plot_xlabel=True, plot_ylabel=True)
plot_one_category_ax1(similar_data, axes[1], plot_xlabel=True, plot_ylabel=True, y_scale=0.3)
plot_one_category_ax2(v2v_data, axes[2], plot_xlabel=True, plot_ylabel=True, left=True)

axes[0].set_title('a) Better V2I Conditions', y=0.0, loc='right', pad=-100, fontsize=30)
axes[0].annotate("Better", fontsize=27, horizontalalignment="center", xy=(0.1, 0.72), xycoords='data',
    xytext=(0.2, 0.64), textcoords='data',
    arrowprops=dict(arrowstyle="->, head_width=0.3", connectionstyle="arc3", lw=3)
)
axes[1].set_title('b) Similar V2I and V2V Conditions' , y=0.0, pad=-100, fontsize=30)
axes[1].annotate("Better", fontsize=25, horizontalalignment="center", xy=(0.36, 0.77), xycoords='data',
    xytext=(0.46, 0.74), textcoords='data',
    arrowprops=dict(arrowstyle="->, head_width=0.3", connectionstyle="arc3", lw=3)
    )
axes[2].set_title('c) Better V2V Conditions', y=0.0, loc='left', pad=-100, fontsize=30)
axes[2].annotate("Better", fontsize=25, horizontalalignment="center", xy=(0.36, 0.74), xycoords='data',
xytext=(0.46, 0.65), textcoords='data',
arrowprops=dict(arrowstyle="->, head_width=0.3", connectionstyle="arc3", lw=3)
)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig('two-dim-acc-latency.pdf')