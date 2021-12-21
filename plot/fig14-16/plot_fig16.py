import matplotlib
import numpy as np
matplotlib.use('AGG')
import matplotlib.pyplot as plt
plt.rc('font', family='sans-serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})

fig = plt.figure(figsize=(7.5, 5.2))
ax = fig.add_subplot(111)
ticks = ['local', 'remote', 'naive', 'Harbor']
ax.bar(0, 0.38590708765539694, alpha=0.7, color='none', ecolor='blue', edgecolor='darkblue', lw=2, hatch='/')
# print(np.mean(local_acc), np.std(local_acc))
ax.errorbar(0, 0.38590708765539694, yerr=0.07983203365173261, capsize=4, color='darkblue')
ax.bar(1, 0.5753014605556781, alpha=0.7, color='none', ecolor='red', edgecolor='red', lw=2, hatch='x')
# print(np.mean(remote_acc), np.std(remote_acc))
ax.errorbar(1, 0.5753014605556781, yerr=0.1302306654578795, capsize=4, color='red')
ax.bar(2, 0.6668036589403477, alpha=0.7, color='none', ecolor='green', edgecolor='green', lw=2, hatch='o')
# print(0.6668036589403477, np.std(combined_naive))
ax.errorbar(2, 0.6668036589403477, yerr=0.131420262221881, capsize=4, color='green')
ax.bar(3, 0.7355364713328815, alpha=0.7, color='none', ecolor='darkorange', edgecolor='darkorange', lw=2, hatch='\\')
# print(0.7355364713328815, np.std(combined_acc))
ax.errorbar(3, 0.7355364713328815, yerr=0.14598966304995303, capsize=4, color='darkorange')
ax.set_xticks(np.arange(4))
ax.set_xticklabels(ticks)
plt.grid(linestyle='--', axis='y', alpha=0.7)
plt.ylabel('Accuracy', fontsize=30)
plt.tight_layout()
plt.savefig('combined-acc.pdf')