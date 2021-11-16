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