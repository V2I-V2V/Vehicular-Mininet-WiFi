import numpy as np
import sys
from ptcl_utils import *
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
matplotlib.rc('font', **{'size': 18})

pred = np.load(sys.argv[1])
truth = np.load(sys.argv[2])
# grid_combine_pred = np.load(sys.argv[3])

grid_pred = calculate_grid_label_before(1, pred)
# grid_pred = calculate_grid_label(1, pred)
grid_truth = calculate_grid_label(1, truth)
# locs = [(-21.642448, 193.882751), (-9.493179, 114.654312)]
# grid_pred = calculate_grid_label_ransac(1, pred, 100, locs[1])
# grid_pred_comb = calculate_grid_label_ransac(1, grid_combine_pred, 100, locs[1])
# grid_truth = calculate_grid_label_ransac(1, truth, 100, locs[1])

mask = grid_truth == 0
indices = grid_truth != 0

# grid_pred[mask] = 0

precision, rst_grid = calculate_precision(grid_pred, grid_truth)
print(precision)
# _, comb_grid = calculate_precision(grid_pred_comb, grid_truth)
# get info w.r.t distances to center

# filtered = rst_grid[25:75, 25:75]
# filtered = rst_grid
# print(len(filtered[filtered == 2]), len(filtered[filtered == 3]))

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ranges = np.arange(10, 50)
# ticks = ['0-10', '10-20', '20-30', '30-40']
fps, fns = [], []
fps_comb, fns_comb = [], []
for r in ranges:
    fp, fn = get_fp_fn_within_range(rst_grid, r)
    # fp_comb, fn_comb = get_fp_fn_within_range(comb_grid, r)
    if len(fps) == 0:
        fps.append(fp)
        fns.append(fn)
    else:
        fps.append(fp-fps[-1])
        fns.append(fn-fns[-1])
    # if len(fps_comb) == 0:
    #     fps_comb.append(fp_comb)
    #     fns_comb.append(fn_comb)
    # else:
    #     fps_comb.append(fp_comb-fps_comb[-1])
    #     fns_comb.append(fn_comb-fns_comb[-1])
ax.plot(ranges, np.array(fps), '--o', label='FP (local)', color='violet')
# ax.plot(ranges, fns, '--o', label='FN (local)', color='limegreen')
# ax.plot(ranges, np.array(fps_comb), '--o', label='FP (remote)', color='purple')
# ax.plot(ranges, fns_comb, '--o', label='FN (remote)', color='green')
# ax.set_xticks(ranges)
# ax.set_xticklabels(ticks)
ax.set_xlabel('Dist to center (m)')
ax.set_ylabel('# of error grids')
ax.legend()


fig, ax = plt.subplots(1, 5, figsize=(30, 6), sharex=True, sharey=True)
ax[0].set_xlim(-55, 55)
ax[0].set_ylim(-55, 55)

for i in range(grid_truth.shape[0]):
    for j in range(grid_truth.shape[1]):
        if grid_truth[i][j] > 0:
            rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1,
                                     edgecolor='darkblue', facecolor='b')
            ax[2].add_patch(rect)
            if grid_pred[i][j] > 0:
                rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1,
                                         edgecolor='darkblue', facecolor='b')
                ax[0].add_patch(rect)
            elif grid_pred[i][j] < 0:
                rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1, edgecolor='maroon',
                                         facecolor='r')
                ax[0].add_patch(rect)
            # if grid_pred_comb[i][j] > 0:
            #     rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1,
            #                              edgecolor='darkblue', facecolor='b')
            #     ax[1].add_patch(rect)
            # else:
            #     rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1, edgecolor='maroon',
            #                              facecolor='r')
            #     ax[1].add_patch(rect)
        elif grid_truth[i][j] < 0:
            rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1, edgecolor='maroon',
                                     facecolor='r')
            ax[2].add_patch(rect)
            if grid_pred[i][j] > 0:
                rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1, edgecolor='darkblue',
                                         facecolor='b')
                ax[0].add_patch(rect)
            elif grid_pred[i][j] < 0:
                rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1, edgecolor='maroon',
                                         facecolor='r')
                ax[0].add_patch(rect)
            # if grid_pred_comb[i][j] > 0:
            #     rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1,
            #                              edgecolor='darkblue', facecolor='b')
            #     ax[1].add_patch(rect)
            # else:
            #     rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1, edgecolor='maroon',
            #                              facecolor='r')
            #     ax[1].add_patch(rect)

for i in range(rst_grid.shape[0]):
    for j in range(rst_grid.shape[1]):
        if rst_grid[i][j] == 2:
            rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1,
                                     edgecolor='purple', facecolor='violet')
            ax[3].add_patch(rect)
        elif rst_grid[i][j] == 3:
            rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1,
                                     edgecolor='green', facecolor='limegreen')
            ax[3].add_patch(rect)

rect = patches.Rectangle((-200, -200), 1, 1, edgecolor='purple',
                         facecolor='violet', label='FP')
ax[3].add_patch(rect)
rect = patches.Rectangle((-200, -200), 1, 1, edgecolor='green',
                         facecolor='limegreen', label='FN')
ax[3].add_patch(rect)
ax[3].legend()

# for i in range(comb_grid.shape[0]):
#     for j in range(comb_grid.shape[1]):
#         if comb_grid[i][j] == 2:
#             rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1,
#                                      edgecolor='purple', facecolor='violet')
#             ax[4].add_patch(rect)
#         elif comb_grid[i][j] == 3:
#             rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1,
#                                      edgecolor='green', facecolor='limegreen')
#             ax[4].add_patch(rect)

plt.show()





# merged_rst = np.load('sample/00_merged.npy')
# merged_rst = calculate_grid_label_before(1, merged_rst)
# updated_grid = combine_merged_results(grid_pred, merged_rst)
# print(calculate_precision(updated_grid, grid_truth))
# print(calculate_precision(merged_rst, grid_truth))
# draw_grids(grid_pred, save_path='./single_vehicle.png')
# draw_grids(updated_grid, save_path='./combine_result.png')
# draw_grids(merged_rst, save_path='./merged_rst.png')
# draw_grids(grid_truth, save_path='./grid_truth.png')
#
#
# # find grids that local detections are correct but merged results are wrong
# known_indices_truth = grid_truth != 0
# local_correct_indices = grid_pred == grid_truth
# merge_incorrect_indices = merged_rst != grid_truth
# desired = np.logical_and(local_correct_indices, merge_incorrect_indices)
# desired = np.logical_and(known_indices_truth, desired)
# np.savetxt('desired.txt', desired, fmt='%s')
# print(len(grid_truth[desired]))
#
# local_incorrect_indices = grid_truth != grid_pred
# merge_correct_indices = merged_rst == grid_truth
# undesired = np.logical_and(local_incorrect_indices, merge_correct_indices)
# undesired = np.logical_and(known_indices_truth, undesired)
# np.savetxt('undesired.txt', undesired, fmt='%s')
# print(len(grid_truth[undesired]))

