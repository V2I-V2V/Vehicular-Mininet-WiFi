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
