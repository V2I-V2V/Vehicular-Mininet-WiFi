import numpy as np
import sys
from ptcl_utils import *
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
matplotlib.rc('font', **{'size': 18})

pred = np.load(sys.argv[1])
truth = np.load(sys.argv[2])

grid_pred = calculate_grid_label_before(1, pred)
# grid_pred = calculate_grid_label(1, pred)
grid_truth = calculate_grid_label(1, truth)

mask = grid_truth == 0
indices = grid_truth != 0

# grid_pred[mask] = 0


print(calculate_precision(grid_pred, grid_truth))

# fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
#
# for i in range(grid_truth.shape[0]):
#     for j in range(grid_truth.shape[1]):
#         if grid_truth[i][j] > 0:
#             rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1,
#                                      edgecolor='darkblue', facecolor='b')
#             ax[1].add_patch(rect)
#             if grid_pred[i][j] > 0:
#
#         elif grid_truth[i][j] < 0:
#             rect = patches.Rectangle((i * 1 - 50, j * 1 - 50), 1, 1, edgecolor='maroon',
#                                      facecolor='r')
#             ax[1].add_patch(rect)

merged_rst = np.load('sample/86_130_mergedpred.npy')
merged_rst = calculate_grid_label_before(1, merged_rst)
updated_grid = combine_merged_results(grid_pred, merged_rst)
print(calculate_precision(updated_grid, grid_truth))
print(calculate_precision(merged_rst, grid_truth))
draw_grids(grid_pred, save_path='./single_vehicle.png')
draw_grids(updated_grid, save_path='./combine_result.png')
draw_grids(merged_rst, save_path='./merged_rst.png')
draw_grids(grid_truth, save_path='./grid_truth.png')


# find grids that local detections are correct but merged results are wrong
known_indices_truth = grid_truth != 0
local_correct_indices = grid_pred == grid_truth
merge_incorrect_indices = merged_rst != grid_truth
desired = np.logical_and(local_correct_indices, merge_incorrect_indices)
desired = np.logical_and(known_indices_truth, desired)
np.savetxt('desired.txt', desired, fmt='%s')
print(len(grid_truth[desired]))

local_incorrect_indices = grid_truth != grid_pred
merge_correct_indices = merged_rst == grid_truth
undesired = np.logical_and(local_incorrect_indices, merge_correct_indices)
undesired = np.logical_and(known_indices_truth, undesired)
np.savetxt('undesired.txt', undesired, fmt='%s')
print(len(grid_truth[undesired]))

