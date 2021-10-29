# -*- coding: utf-8 -*-
# Filename : visualization
__author__ = 'Ruiyang Zhu'

import matplotlib
import numpy as np
import open3d as o3d
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

matplotlib.rc('font', **{'size': 18})

parser = argparse.ArgumentParser()
parser.add_argument("--data", default='pred.npy', help="point cloud data to visualize (in npy format)")
parser.add_argument("--make_undefined_undrivable", default=False, action='store_true', help="mark undefined grid as "
                                                                                          "occupied")
args = parser.parse_args()
data_file = args.data
make_undefined_occupied = args.make_undefined_undrivable

pcl = np.load(data_file)

print(pcl.shape[0])

pcl_ground = pcl[pcl[:, 3] == 1]
pcl_object = pcl[pcl[:, 3] == 0]
# print(pcl_ground[:, 0].min())
# print(pcl_ground[:, 0].max())
# print(pcl_ground[:, 1].min())
# print(pcl_ground[:, 1].max())

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
pcd.paint_uniform_color([1, 0, 0])
# np_colors = np.copy(pcl[:, :3])
pcd_gnd = o3d.geometry.PointCloud()
pcd_gnd.points = o3d.utility.Vector3dVector(pcl_ground[:, :3])
pcd_gnd.paint_uniform_color([0, 0, 1])


# pcd.colors = o3d.utility.Vector3dVector(np_colors)
o3d.visualization.draw_geometries([pcd, pcd_gnd])


def draw_grids(points):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(points[:, 0], points[:, 1], s=0.1, c="blue")
    ticks = np.arange(-100, 100, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(color='r', linestyle='--')
    plt.show()


def process_grid(grid_size, points):
    x_size, y_size = int(100 / grid_size), int(100 / grid_size)

    grid = np.zeros((x_size, y_size), dtype=int)
    existence = np.zeros((x_size, y_size), dtype=int)
    x_start_idx, y_start_idx = 0, 0

    for point in points:
        x_idx, y_idx = int((point[0] + 50) / grid_size), int((point[1] + 50) / grid_size)
        if 100 > x_idx >= 0 and 100 > y_idx >= 0:
            if point[3] != 1:
                # obj
                grid[x_idx][y_idx] -= 1
                existence[x_idx][y_idx] = 1
            elif point[3] == 1:
                grid[x_idx][y_idx] += 1
                existence[x_idx][y_idx] = 1

    fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True, sharey=True)
    ax[0].scatter(points[:, 0], points[:, 1], s=0.1)

    minor_ticks = np.arange(-50, 55, 1)
    # ax[1].set_xticks(ticks)
    ax[0].set_xlim(-55, 55)
    ax[0].set_ylim(-55, 55)

    ax[0].set_ylabel('Original Point Cloud')
    ax[0].set_xticks(minor_ticks, minor=True)
    ax[0].set_yticks(minor_ticks, minor=True)

    # plot segmented point clouds (mark ground points)
    ax[1].scatter(pcl_object[:, 0], pcl_object[:, 1], s=0.1, c='maroon')
    ax[1].scatter(pcl_ground[:, 0], pcl_ground[:, 1], s=0.11, c='cyan')
    ax[1].scatter(-100, -100, c='maroon', label='Object')
    ax[1].scatter(-100, -100, c='cyan', label='Ground')
    ax[1].legend(fontsize=12, loc='lower left')
    ax[1].set_ylabel('Ground Extraction')

    # np.savetxt('pred.txt', existence[45:55, 45:55], fmt='%d')

    cnt = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i][j] > 0:
                rect = patches.Rectangle((i * grid_size - 50, j * grid_size - 50), grid_size, grid_size,
                                         edgecolor='darkblue', facecolor='b')
                ax[2].add_patch(rect)
                cnt += 1
            elif grid[i][j] < 0:
                rect = patches.Rectangle((i * grid_size - 50, j * grid_size - 50), grid_size, grid_size,
                                         edgecolor='maroon', facecolor='r')
                ax[2].add_patch(rect)
                cnt += 1
            elif make_undefined_occupied:
                rect = patches.Rectangle((i * grid_size - 50, j * grid_size - 50), grid_size, grid_size,
                                         edgecolor='maroon', facecolor='r')
                ax[2].add_patch(rect)
                cnt += 1

    print(cnt)

    rect = patches.Rectangle((-200, -200), grid_size, grid_size, edgecolor='maroon',
                             facecolor='r', label='Undrivable')
    ax[2].add_patch(rect)
    rect = patches.Rectangle((-200, -200), grid_size, grid_size, edgecolor='darkblue',
                             facecolor='b', label='Drivable')
    ax[2].add_patch(rect)
    if not make_undefined_occupied:
        rect = patches.Rectangle((200, 200), grid_size, grid_size, edgecolor='grey',
                                 facecolor='white', label='Undefined')
        ax[2].add_patch(rect)
    ax[2].legend(fontsize=12, loc='lower left')
    ax[2].set_ylabel('Drivable Space Map')
    ax[2].grid(linestyle='-', which='minor', alpha=0.4)
    ax[2].grid(linestyle='-', which='major', alpha=0.4)
    plt.tight_layout()
    plt.show()


# plt.savefig('gta.pdf')


process_grid(1, pcl)
# draw_grids(pcl_ground)
