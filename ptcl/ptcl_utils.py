import numpy as np
import open3d as o3d
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


def read_ptcl_data(ptcl_fname):
    """ Read point cloud data (.bin/.pcd/.npy) to numpy array

    Args:
        ptcl_fname ([str]): ptcl file name to process
    """
    if '.bin' in ptcl_fname:
        ptcl = np.fromfile(ptcl_fname, dtype=np.float32, count=-1).reshape([-1, 4])
    elif '.pcd' in ptcl_fname:
        ptcl = o3d.io.read_point_cloud(ptcl_fname)
        ptcl = np.asarray(ptcl.points)
    elif '.npy' in ptcl_fname:
        ptcl = np.load(ptcl_fname)
    return ptcl


def draw_ptcl(pointcloud, mode='3d', show=True, save=False, save_path=None):
    if mode == '3d':
        draw_3d(pointcloud, show, save, save_path)
    elif mode == '2d':
        draw_2d(pointcloud, show, save, save_path)


def draw_3d(pointcloud, show=True, save=False, save_path=None):
    """Draw point cloud in 3D dimention using open3d

    Args:
        pointcloud ([numpy array]): ptcl data in numpy format
        show (bool, optional): whether to show the figure. Defaults to True.
        save (bool, optional): whether to save the figure. Defaults to False.
        save_path ([type], optional): figure save path when save is True. Defaults to None.
    """
    # TODO: paint different pcd in various colors? Currently the merge function will produce
    # point clouds from all vehicles
    pointcloud_all = o3d.geometry.PointCloud()
    pointcloud_all.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
    pointcloud_all.paint_uniform_color([0, 0, 1])
    if show:
        o3d.visualization.draw_geometries([pointcloud_all])
    if save:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pointcloud_all)
        vis.update_geometry(pointcloud_all)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1)
        vis.capture_screen_image(save_path, True)
        vis.destroy_window()


def draw_2d(data, show=True, save=False, save_path=None):
    """Draw point cloud in 3D dimention using matplotlib

    Args:
        data ([np array]): ptcl data in numpy format
        show (bool, optional): whether to show the figure. Defaults to True.
        save (bool, optional): whether to save the figure. Defaults to False.
        save_path ([type], optional): figure save path when save is True. Defaults to None.
    """
    plt.scatter(data[:, 0], data[:, 1], s=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(save_path)


def draw_grids(grid, grid_size=1, make_undefined_occupied=False, save_path='detection_grid.png'):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    minor_ticks = np.arange(-50, 55, 1)
    # ax[1].set_xticks(ticks)
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)

    ax.set_ylabel('Original Point Cloud')
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    cnt = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i][j] > 0:
                rect = patches.Rectangle((i * grid_size - 50, j * grid_size - 50), grid_size, grid_size,
                                         edgecolor='darkblue', facecolor='b')
                ax.add_patch(rect)
                cnt += 1
            elif grid[i][j] < 0:
                rect = patches.Rectangle((i * grid_size - 50, j * grid_size - 50), grid_size, grid_size,
                                         edgecolor='maroon', facecolor='r')
                ax.add_patch(rect)
                cnt += 1
            elif make_undefined_occupied:
                rect = patches.Rectangle((i * grid_size - 50, j * grid_size - 50), grid_size, grid_size,
                                         edgecolor='maroon', facecolor='r')
                ax.add_patch(rect)
                cnt += 1

    rect = patches.Rectangle((-100, -100), grid_size, grid_size, edgecolor='maroon',
                             facecolor='r', label='Undrivable')
    ax.add_patch(rect)
    rect = patches.Rectangle((-100, -100), grid_size, grid_size, edgecolor='darkblue',
                             facecolor='b', label='Drivable')
    ax.add_patch(rect)
    if not make_undefined_occupied:
        rect = patches.Rectangle((100, 100), grid_size, grid_size, edgecolor='grey',
                                 facecolor='white', label='Undefined')
    ax.add_patch(rect)
    ax.legend(fontsize=12, loc='lower left')
    ax.set_ylabel('Drivable Space Map')
    ax.grid(linestyle='-', which='minor', alpha=0.4)
    ax.grid(linestyle='-', which='major', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)


def get_number_of_points_in_region(ptcl, x_center, y_center, x_width, y_width):
    """ Get # of points in a region (grid)

    Args:
        ptcl ([np.array]): ptcl data in numpy arr
        x_center ([float]): x center of grid
        y_center ([float]): y center of grid
        x_width ([float]): width of grid in x
        y_width ([float]): width of grid in y

    Returns:
        [int]: number of points inside grid
    """
    filtered = ptcl[ptcl[:, 0] > x_center - x_width / 2]
    filtered = filtered[filtered[:, 0] < x_center + x_width / 2]
    filtered = filtered[filtered[:, 1] > y_center - y_width / 2]
    filtered = filtered[filtered[:, 1] < y_center + y_width / 2]
    return filtered.shape[0]


def save_ptcl(ptcl, save_path, format='bin'):
    """ Save ptcl (numpy array) to desired format

    Args:
        ptcl ([numpy array]): point cloud data in numpy format
        save_path ([str]): path to save ptcl
        format (str, optional): format to save. Defaults to 'bin'.
    """
    if ptcl.shape[1] == 3:
        ptcl = np.hstack([ptcl, np.ones((ptcl.shape[0], 1), dtype=np.float32)])
    if format == 'bin':
        with open(save_path, 'w') as f:
            ptcl.tofile(f)
    elif format == 'pcd':
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(ptcl[:, :3])
        o3d.io.write_point_cloud(save_path, pointcloud)
    elif format == 'npy':
        np.save(save_path, ptcl)


def calculate_grid_label(grid_size, points):
    x_size, y_size = int(100 / grid_size), int(100 / grid_size)
    grid = np.zeros((x_size, y_size), dtype=int)
    for point in points:
        x_idx, y_idx = int((point[0] + 50) / grid_size), int((point[1] + 50) / grid_size)
        if x_idx < 100 and y_idx < 100:
            if point[3] == 0:
                # obj
                grid[x_idx][y_idx] -= 1
            elif point[3] == 1:
                # ground
                grid[x_idx][y_idx] += 1

    grid[grid > 0] = 1  # drivable
    grid[grid < 0] = -1  # object
    # print(len(grid[grid > 0]))

    return grid


def calculate_grid_label_ransac(grid_size, points, space_width=100, center=(0, 0)):
    x_size, y_size = int(space_width / grid_size), int(space_width / grid_size)
    grid = np.zeros((x_size, y_size), dtype=int)
    for point in points:
        x_idx, y_idx = int((point[0] - center[0] + space_width / 2) / grid_size), \
                       int((point[1] - center[1] + space_width / 2) / grid_size)
        if space_width > x_idx >= 0 and space_width > y_idx >= 0:
            if point[3] != 1:
                # obj
                grid[x_idx][y_idx] -= 1
            elif point[3] == 1:
                # ground
                grid[x_idx][y_idx] += 1
    grid[grid > 0] = 1  # drivable
    grid[grid < 0] = -1  # object
    return grid


def calculate_grid_label_before(grid_size, points):
    x_size, y_size = int(100 / grid_size), int(100 / grid_size)
    grid = np.zeros((x_size, y_size), dtype=int)
    for point in points:
        x_idx, y_idx = int((point[0] + 50) / grid_size), int((point[1] + 50) / grid_size)
        if x_idx < 100 and y_idx < 100:
            if point[3] == 1:
                # obj
                grid[x_idx][y_idx] -= 1
            elif point[3] == 0:
                # ground
                grid[x_idx][y_idx] += 1

    grid[grid > 0] = 1  # drivable
    grid[grid < 0] = -1  # object
    # print(len(grid[grid > 0]))

    return grid


def calculate_precision(grid_pred, grid_truth):
    result_grid = np.zeros(grid_truth.shape)
    correct, total = 0, 0

    TP, FP, FN, TN = 0, 0, 0, 0

    for x_idx in range(grid_truth.shape[0]):
        for y_idx in range(grid_truth.shape[1]):
            if grid_truth[x_idx][y_idx] != 0:
                if grid_pred[x_idx][y_idx] == grid_truth[x_idx][y_idx]:
                    correct += 1
                total += 1
                if grid_truth[x_idx][y_idx] == 1 and grid_pred[x_idx][y_idx] != 1:
                    FN += 1
                    result_grid[x_idx][y_idx] = 3  # FN
                elif grid_truth[x_idx][y_idx] != 1 and grid_pred[x_idx][y_idx] == 1:
                    FP += 1
                    result_grid[x_idx][y_idx] = 2  # FP
                elif grid_truth[x_idx][y_idx] == 1 and grid_pred[x_idx][y_idx] == 1:
                    TP += 1
                    result_grid[x_idx][y_idx] = 1  # TP
                elif grid_truth[x_idx][y_idx] == -1 and grid_pred[x_idx][y_idx] == -1:
                    TN += 1
                    result_grid[x_idx][y_idx] = 4  # TN
    precision = correct / total

    pre = TP / (TP + FP)
    recall = TP / (TP + FN)

    print(pre, recall)
    print(TP, FP, FN, TN, TP + FP + FN + TN, (TP + TN) / (TP + FP + FN + TN))

    # rst = grid_pred == grid_truth
    # # precision
    # precision = len(rst[rst == True])/grid_truth.size

    return precision, result_grid


def combine_merged_results(grid_single, grid_merged):
    """combine merged results to single-vehicle detection

    Args:
        grid_single ([numpy.array]): grid label of single vehicle detection
        grid_merged ([numpy.array]): grid label of multi vehicle merged detection

    Returns:
        [numpy.array]: updated detection results
    """
    grid_updated = np.copy(grid_single)
    # find the grid label that is unknown
    unknown_indices = grid_single == 0
    grid_updated[unknown_indices] = grid_merged[unknown_indices]
    return grid_updated


def get_fp_fn_within_range(result_grid, filter_range):
    filtered = result_grid[int(result_grid.shape[0] / 2 - filter_range):int(result_grid.shape[0] / 2 + filter_range)] \
        [int(result_grid.shape[1] / 2 - filter_range):int(result_grid.shape[1] / 2 + filter_range)]
    fp, fn = len(filtered[filtered == 2]), len(filtered[filtered == 3])
    return fp, fn


def plot_vehicle_location(vehicle_locs, vehicle_ids):
    for i in range(len(vehicle_locs)):
        loc = vehicle_locs[i]
        plt.scatter(loc[0], loc[1], label='Vehicle%s' % vehicle_ids[i])
    plt.legend()
    plt.show()


def get_GndSeg(sem_label, GndClasses):
    index = np.isin(sem_label, GndClasses)
    GndSeg = np.ones(sem_label.shape)
    GndSeg[index] = 1
    index = np.isin(sem_label, [0, 1])
    GndSeg[index] = 0
    return GndSeg


def concat_labels(sem_labels):
    merged = np.concatenate(sem_labels)
    return merged


def plot_grid(grid, grid_size):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(-int(grid.shape[0]/grid_size/2)-5, int(grid.shape[0]/grid_size/2)+5)
    ax.set_ylim(-int(grid.shape[1]/grid_size/2)-5, int(grid.shape[1]/grid_size/2)+5)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i][j] > 0:
                rect = patches.Rectangle(((i-grid.shape[0]/2) * grid_size, (j-grid.shape[1]/2) * grid_size), grid_size, grid_size,
                                         edgecolor='darkblue', facecolor='b')
                ax.add_patch(rect)
            elif grid[i][j] < 0:
                rect = patches.Rectangle(((i-grid.shape[0]/2) * grid_size, (j-grid.shape[1]/2) * grid_size), grid_size, grid_size,
                                         edgecolor='maroon', facecolor='r')
                ax.add_patch(rect)

    rect = patches.Rectangle((-200, -200), grid_size, grid_size, edgecolor='maroon',
                             facecolor='r', label='Undrivable')
    ax.add_patch(rect)
    rect = patches.Rectangle((-200, -200), grid_size, grid_size, edgecolor='darkblue',
                             facecolor='b', label='Drivable')
    ax.add_patch(rect)
    ax.set_ylabel('Drivable Space Map')
    plt.legend()
    plt.tight_layout()
    plt.show()

## TODO: Transform the ptcl to a global reference?
