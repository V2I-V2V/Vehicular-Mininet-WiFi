# -*- coding: utf-8 -*-
# Filename : visualization

import numpy as np
import open3d as o3d
import sys

filename = sys.argv[1]
if '.bin' in filename:
	ptcl = np.fromfile(sys.argv[1], dtype=np.float32, count=-1).reshape([-1,4])
elif '.pcd' in filename:
	ptcl = o3d.io.read_point_cloud(filename)
	ptcl = np.asarray(ptcl.points)
elif '.npy' in filename:
	ptcl = np.load(filename)
	

count = 0
print(ptcl.shape[0])
for p in ptcl:
	if np.isnan(p[0]):
		count = count + 1
# print(count)
ptcl = ptcl[np.logical_not(np.isnan(ptcl[:,0]))]
print(ptcl.shape[0])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ptcl[:,:3])
pcd.paint_uniform_color([0, 0, 1])
o3d.visualization.draw_geometries([pcd])


# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)
# vis.update_geometry(pcd)
# vis.poll_events()
# vis.update_renderer()
# vis.capture_screen_image(path)
# vis.destroy_window()
