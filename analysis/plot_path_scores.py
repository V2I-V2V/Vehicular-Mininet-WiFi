import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
font = {'family' : 'DejaVu Sans',
        'size'   : 18}
matplotlib.rc('font', **font)
import statistics as stat

dist_combined = [0.3685444679663241, 0.14389321881345254]
bw_combined = [0.85, 0.8875]
intf_combined = [0.6666666666666667, 0.8333333333333334]
dist_combined_min = [0.30389321881345254, 0.5537864045000421]
bw_combined_min = [0.7875, 0.9125]
intf_combined_min = [0.4, 0.8333333333333334]


harmonic_mean_score = [stat.harmonic_mean(dist_combined)+stat.harmonic_mean(bw_combined)+stat.harmonic_mean(intf_combined),
    stat.harmonic_mean(dist_combined_min)+stat.harmonic_mean(bw_combined_min)+stat.harmonic_mean(intf_combined_min)]

min_mean_score = [dist_combined[0]+bw_combined[0]+intf_combined[0],\
                    dist_combined[1]+bw_combined[1]+intf_combined[1],\
                    dist_combined_min[0]+bw_combined_min[0]+intf_combined_min[0],\
                    dist_combined_min[1]+bw_combined_min[1]+intf_combined_min[1]]



fig, ax = plt.subplots(1, 1, figsize=(9,4))
cnt = 0
ax.bar(np.arange(cnt, cnt+2), dist_combined, color='r', alpha=0.5, label='assignment 1')
cnt += 2
ax.bar(np.arange(cnt, cnt+2), dist_combined_min, color='b', alpha=0.5, label='assignment 2')
cnt += 4
ax.bar(np.arange(cnt, cnt+2),bw_combined, color='r', alpha=0.5)
cnt += 2
ax.bar(np.arange(cnt, cnt+2), bw_combined_min, color='b', alpha=0.5)
cnt += 4
ax.bar(np.arange(cnt, cnt+2), intf_combined, color='r', alpha=0.5)
cnt += 2
ax.bar(np.arange(cnt, cnt+2), intf_combined_min, color='b', alpha=0.5)
cnt += 4
ax.bar(np.arange(cnt, cnt+1), harmonic_mean_score[0], color='r', alpha=0.5)
ax.bar(np.arange(cnt+1, cnt+2), harmonic_mean_score[1], color='b', alpha=0.5)
# cnt += 4
# ax.bar(np.arange(cnt, cnt+2), min_mean_score[:2], color='r', alpha=0.5)
# ax.bar(np.arange(cnt+2, cnt+4), min_mean_score[2:], color='b', alpha=0.5)


ax.legend()
ax.set_xticks([1, 8, 14, 19])
ax.set_xticklabels(['dist-scores', 'bw-scores', 'intf-scores', 'harmonic\nmean'])
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('score-hist.pdf')


score = [0.10194919928351012,0.7916666666666665, 0.8888888888888888]
score_2 = [0.6753178248957521, 0.31721311475409836, 0.6896551724137931]
fig, ax = plt.subplots(1, 1, figsize=(6,4))
cnt = 0
plt.bar(np.arange(cnt, cnt+1), score[0], color='r', alpha=0.5, ecolor='maroon', label='assignment 1')
ax.bar(np.arange(cnt+1, cnt+2), score_2[0], color='b', alpha=0.5, ecolor='darkblue',linewidth=2, label='assignment 2')
cnt += 4
ax.bar(np.arange(cnt, cnt+1), score[1], color='r', alpha=0.5,  ecolor='maroon')
ax.bar(np.arange(cnt+1, cnt+2), score_2[1], color='b', alpha=0.5, ecolor='darkblue')
cnt += 4
ax.bar(np.arange(cnt, cnt+1), score[2], color='r', alpha=0.5,  ecolor='maroon')
ax.bar(np.arange(cnt+1, cnt+2), score_2[2], color='b', alpha=0.5, ecolor='darkblue')
cnt += 4
ax.bar(np.arange(cnt, cnt+1), sum(score),  color='r', alpha=0.5, ecolor='maroon')
ax.bar(np.arange(cnt+1, cnt+2), sum(score_2), color='b', alpha=0.5, ecolor='darkblue')
ax.set_xticks([0.5, 4.5, 8.5, 12.5])
ax.set_xticklabels(['distance\nscores', 'bw\nscores', 'intference\nscores', 'sum'])
ax.legend()
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('score-factor.pdf')


score = [0.0194919928351012,0.7916666666666665,0.8888888888888888]
score_old = [0.6753178248957521,0.0019774011299435027, 0.6896551724137931] 

score_bad_ass = [1.7491714215057326, 1.8241714215057323, 1.7825047548390653, 1.7075047548390656, 1.7075047548390656, 1.6875047548390656, 1.7075047548390656, 1.7991714215057324, 1.7993714215057324]
scores_old_ass = [1.9394410824159283, 1.5657082914271923, 1.3669503984394886, 1.7478811605748514, 1.7478811605748514, 1.7778811605748514, 1.7478811605748514, 2.319221363322617, 2.319221363322617]

fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.plot(np.arange(len(score_bad_ass)), score_bad_ass, '--o', label='suboptimal-assignment score')
ax.plot(np.arange(len(score_bad_ass)), scores_old_ass, '-^', label='best assignment score')
ax.set_xticks(np.arange(len(score_bad_ass)))
ax.set_xticklabels(np.arange(15, 15+len(score_bad_ass)))
Q = ax.quiver(0.7, score_bad_ass[3] + 0.21, 2.5, -0.7, angles="xy", color='darkgreen',
                      width=0.004, headwidth=3, headlength=3, zorder=8)
Q = ax.quiver(6.7, scores_old_ass[7] + 0.11, 5.0, -1.7, angles="xy", color='darkgreen',
                      width=0.004, headwidth=3, headlength=3, zorder=8)
ax.quiverkey(Q, 0.25, 0.68, 6, label='Assignment Change', labelpos='E', labelcolor='darkgreen', fontproperties=font)
plt.xlabel('Time (s)')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.savefig('timeseris-ass.png')

