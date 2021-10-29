import numpy as np
import ptcl_utils
from ground import my_ransac_v5

label1 = np.fromfile('/home/ryanzhu/V2I+V2V/Carla/labels/86/1000.label', dtype=np.uint32)
label1.astype(np.uint16)
label2 = np.fromfile('/home/ryanzhu/V2I+V2V/Carla/labels/130/1000.label', dtype=np.uint32)
label2.astype(np.uint16)

merged_label = ptcl_utils.concat_labels([label1])

print(merged_label.shape)

ptcl = ptcl_utils.read_ptcl_data('/home/ryanzhu/MultiVehiclePerception/86_130_merged.bin')
ptcl = ptcl_utils.read_ptcl_data('/home/ryanzhu/V2I+V2V/Carla/lidar/86/1000.npy')
ptcl[:, 3] = 1
trans = np.load('/home/ryanzhu/V2I+V2V/Carla/lidar/86/1000.trans.npy')
ptcl = np.dot(trans, ptcl[:, :4].T).T

ptcl2 = ptcl_utils.read_ptcl_data('/home/ryanzhu/V2I+V2V/Carla/lidar/130/1000.npy')
ptcl2[:, 3] = 1
trans2 = np.load('/home/ryanzhu/V2I+V2V/Carla/lidar/130/1000.trans.npy')
ptcl2 = np.dot(trans2, ptcl2[:, :4].T).T
GndSeg = ptcl_utils.get_GndSeg(merged_label, GndClasses=[40, 44, 48, 49, 60, 72])
GndSeg2 = ptcl_utils.get_GndSeg(label2, GndClasses=[40, 44, 48, 49, 60, 72])
ptcl2[:, 3] = GndSeg2
print(GndSeg.shape)
points_truth_labeled = np.copy(ptcl)
points_truth_labeled[:, 3] = GndSeg
merged = np.vstack([points_truth_labeled, ptcl2])
np.save('86_130_gt.npy', merged)

# ptcl_utils.calculate_grid_label_ransac(1, merged, 100, )

gt_ptcl = ptcl_utils.read_ptcl_data('86_130_gt.npy')
pred1 = ptcl_utils.read_ptcl_data('1000_86_pred.npy')
pred2 = ptcl_utils.read_ptcl_data('1000_130_pred.npy')
locs = [(-21.642448, 193.882751), (-9.493179, 114.654312)]
grid_pred_single = [ptcl_utils.calculate_grid_label_ransac(1, pred1, 100, locs[0])[0],
                    ptcl_utils.calculate_grid_label_ransac(1, pred2, 100, locs[1])[0]]
truth_grids = []
for i in range(len(locs)):
    grid = ptcl_utils.calculate_grid_label_ransac(1, gt_ptcl, 100, locs[i])[0]
    truth_grids.append(grid)
    precision, rst_grid = ptcl_utils.calculate_precision(grid_pred_single[i], grid)
    print('precision ', precision)

merged_pred = ptcl_utils.read_ptcl_data('/home/ryanzhu/MultiVehiclePerception/86_130_merged_10_8.bin')
p2, p1, best_model, _ = my_ransac_v5(merged_pred, 10000, P=0.8, distance_threshold=0.35,
                                     lidar_height=-2.03727 + 0.1, lidar_height_down=-2.03727 - 0.1,
                                     use_all_sample=True)
merged_pred[:, 3] = 0  # object
merged_pred[p2, 3] = 1  # ground
np.save('86_130_comb_pred.npy', merged_pred)
merged_detect_grid = []
for i in range(len(locs)):
    grid = ptcl_utils.calculate_grid_label_ransac(1, merged_pred, 100, locs[i])[0]
    merged_detect_grid.append(grid)
    precision, rst_grid = ptcl_utils.calculate_precision(grid, truth_grids[i])
    ptcl_utils.plot_grid(grid, 1)
    ptcl_utils.plot_grid(truth_grids[i], 1)
    print('precision ', precision)

print('--------------')
for i in range(len(locs)):
    combined_grid = ptcl_utils.combine_merged_results(grid_pred_single[i], merged_detect_grid[i])
    precision, rst_grid = ptcl_utils.calculate_precision(combined_grid, truth_grids[i])
    print('precision ', precision)
