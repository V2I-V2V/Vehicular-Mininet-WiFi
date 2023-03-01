import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

# get colors
cmap20 = plt.cm.tab20  # define the colormap
# extract all colors from the .jet map
colorlist20 = [cmap20(i) for i in range(cmap20.N)]

plt.rc('font', family='sans-serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})

# ctrl msg latency w/o data msg
ctrl_wo_data_p2p = np.loadtxt('data-12012050/node_0_dl_latency.txt')
ctrl_wo_data_cellular = np.loadtxt('data-12012050/node_1_dl_latency.txt')

# ctrl msg latency w/ data msg
ctrl_w_data_p2p = np.loadtxt('data-12012135/node_0_dl_latency.txt')
ctrl_w_data_cellular = np.loadtxt('data-12012135/node_1_dl_latency.txt')


print(np.median(ctrl_wo_data_p2p), np.median(ctrl_w_data_p2p))

fig, axes = plt.subplots(1, 1, figsize=(5, 5))

b1 = axes.boxplot(ctrl_wo_data_p2p, positions=[0], whis=(5, 95), 
                  autorange=True, showfliers=False, widths=0.25, patch_artist=True)
b2 = axes.boxplot(ctrl_w_data_p2p, positions=[1.5], whis=(5, 95), 
                  autorange=True, showfliers=False, widths=0.25, patch_artist=True)
b1['boxes'][0].set(color=colorlist20[2])
b1['boxes'][0].set(linewidth=1)
b1['boxes'][0].set(facecolor = colorlist20[3])
b1['boxes'][0].set(label='P2P WiFi Network')
for cap in b1['caps']:
        cap.set(color=colorlist20[2], linewidth=1)
b1['medians'][0].set(color=colorlist20[2], linewidth=1)
for whisker in b1['whiskers']:
        whisker.set(color=colorlist20[2], linewidth=1.5, linestyle=':')

b2['boxes'][0].set(color=colorlist20[2])
b2['boxes'][0].set(linewidth=1)
b2['boxes'][0].set(facecolor = colorlist20[3])
b2['boxes'][0].set(label='P2P WiFi Network')
for cap in b2['caps']:
        cap.set(color=colorlist20[2], linewidth=1)
b2['medians'][0].set(color=colorlist20[2], linewidth=1)
for whisker in b2['whiskers']:
        whisker.set(color=colorlist20[2], linewidth=1.5, linestyle=':')
        

b3 = axes.boxplot(ctrl_wo_data_cellular, positions=[0.3], whis=(5, 95), 
                  autorange=True, showfliers=False, widths=0.25, patch_artist=True)
b4 = axes.boxplot(ctrl_w_data_cellular, positions=[1.8], whis=(5, 95), 
                  autorange=True, showfliers=False, widths=0.25, patch_artist=True)
b3['boxes'][0].set(color=colorlist20[0])
b3['boxes'][0].set(linewidth=1)
b3['boxes'][0].set(facecolor = colorlist20[1])
b3['boxes'][0].set(hatch='//')
b3['boxes'][0].set(label='Wired Network')
for cap in b3['caps']:
        cap.set(color=colorlist20[0], linewidth=1)
b3['medians'][0].set(color=colorlist20[0], linewidth=1)
for whisker in b3['whiskers']:
        whisker.set(color=colorlist20[0], linewidth=1.5, linestyle=':')
b4['boxes'][0].set(color=colorlist20[0])
b4['boxes'][0].set(linewidth=1)
b4['boxes'][0].set(facecolor = colorlist20[1])
b4['boxes'][0].set(hatch='//')
b4['boxes'][0].set(label='Wired Network')
for cap in b4['caps']:
        cap.set(color=colorlist20[0], linewidth=1)
b4['medians'][0].set(color=colorlist20[0], linewidth=1)
for whisker in b4['whiskers']:
        whisker.set(color=colorlist20[0], linewidth=1.5, linestyle=':')
        
axes.set_xticks([0.15, 1.65])
axes.set_xticklabels(['w/o concurrent\ndata', 'w/ concurrent\ndata'])
axes.set_ylabel("Control Msg Latency (s)")
# axes.set_ylim([0, 0.1])

axes.legend([b1["boxes"][0], b3["boxes"][0]], ['P2P WiFi Network', 'Wired Network'], fontsize=15, facecolor='whitesmoke',
            loc='upper left')
plt.grid(linestyle='--', axis='y')
plt.tight_layout()
plt.savefig('fig6.pdf')