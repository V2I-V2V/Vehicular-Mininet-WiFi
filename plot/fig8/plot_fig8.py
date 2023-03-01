import matplotlib
import numpy as np
matplotlib.use('AGG')
import matplotlib.pyplot as plt
plt.rc('font', family='sans-serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})
import json

f=open('acc_dict.json')
qb_to_acc = json.load(f)
# qb_range = np.arange(13, 7)
qb_range = np.array([12, 11, 10, 9, 8, 7])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
cnt = 0
# ax.boxplot(qb_to_size_dict['raw'], positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
# cnt += 1
for qb, sizes in qb_to_acc.items():
    if qb != 'raw':
        ax.boxplot(sizes, positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
        cnt += 1
tick_labels = []
for qb in qb_range:
    tick_labels.append(str(qb))
ax.set_xticks(np.arange(len(qb_to_acc.keys())))
ax.set_xticklabels(tick_labels)
ax.set_ylabel('Drivable Space\nDetection Accuracy', loc='top')
ax.set_xlabel('Quantization Bits', loc='right')
ax.grid(linestyle='--', axis='y')
plt.tight_layout()
plt.savefig('acc_vs_compression_qb.pdf')