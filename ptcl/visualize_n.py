# -*- coding: utf-8 -*-
# Filename : visualization_n
__author__ = 'Xumiao Zhang'

import numpy as np
import open3d as o3d
import sys
import torch
from math import sin, cos

ptcl_path1 = '../DeepGTAV-data/object-0227-1/'
oxts_path1 = '../DeepGTAV-data/object-0227-1/oxts/'
ptcl_path2 = '../DeepGTAV-data/object-0227-1/alt_perspective/0022786/'
oxts_path2 = '../DeepGTAV-data/object-0227-1/alt_perspective/0022786/oxts/'
ptcl_path3 = '../DeepGTAV-data/object-0227-1/alt_perspective/0037122/'
oxts_path3 = '../DeepGTAV-data/object-0227-1/alt_perspective/0037122/oxts/'
ptcl_path4 = '../DeepGTAV-data/object-0227-1/alt_perspective/0191023/'
oxts_path4 = '../DeepGTAV-data/object-0227-1/alt_perspective/0191023/oxts/'
ptcl_path5 = '../DeepGTAV-data/object-0227-1/alt_perspective/0399881/'
oxts_path5 = '../DeepGTAV-data/object-0227-1/alt_perspective/0399881/oxts/'
ptcl_path6 = '../DeepGTAV-data/object-0227-1/alt_perspective/0735239/'
oxts_path6 = '../DeepGTAV-data/object-0227-1/alt_perspective/0735239/oxts/'


def rotate(oxts1, oxts2):
	### transformation matrix - rotation (to the perspective of oxts1)
	dYaw = oxts2[5] - oxts1[5]
	dPitch = oxts2[4] - oxts1[4]
	dRoll = oxts2[3] - oxts1[3]
	rotation_Z = torch.tensor([[cos(dYaw), -sin(dYaw), 0, 0], [sin(dYaw), cos(dYaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	rotation_Y = torch.tensor([[cos(dPitch), 0, sin(dPitch), 0], [0, 1, 0, 0], [-sin(dPitch), 0, cos(dPitch), 0], [0, 0, 0, 1]])
	rotation_X = torch.tensor([[1, 0, 0, 0], [0, cos(dRoll), -sin(dRoll), 0], [0, sin(dRoll), cos(dRoll), 0], [0, 0, 0, 1]])
	rotation = torch.mm(torch.mm(rotation_Z, rotation_Y), rotation_X)
	return rotation  # .numpy()

def translate(oxts1, oxts2):
	### transformation matrix - translation (to the perspective of oxts1)
	da = oxts2[0] - oxts1[0]  # south --> north
	db = oxts2[1] - oxts1[1]  # east --> west
	dx = da * cos(oxts1[5]) + db * sin(oxts1[5])
	dy = da * -sin(oxts1[5]) + db * cos(oxts1[5])
	dz = oxts2[2] - oxts1[2]
	translation = torch.tensor([dx, dy, dz, 0])
	return translation

def convert(points_oxts_primary, points_oxts_secondary):
	points_0, oxts_0 = points_oxts_primary  # primary vehicle
	# pcl = points_0
	pcl = torch.tensor(points_0)  # torch.from_numpy(points_0)
	for (points, oxts) in points_oxts_secondary:
		points = torch.tensor(points)  # torch.from_numpy(points)
		rotation = rotate(oxts_0, oxts)
		translation = translate(oxts_0, oxts).repeat(np.shape(points)[0],1)  # .numpy()
		# pcl = np.float32(np.concatenate((pcl, np.matmul(points, np.transpose(rotation.numpy())) + translation.numpy())))
		pcl_trasformed = torch.mm(points, torch.t(rotation)) + translation
	return pcl_trasformed.numpy()


if __name__ == '__main__':

	pcls = []
	pcl1 = np.memmap(ptcl_path1+sys.argv[1]+'.bin', dtype='float32', mode='r').reshape([-1,4])
	pcl2 = np.memmap(ptcl_path2+sys.argv[1]+'.bin', dtype='float32', mode='r').reshape([-1,4])
	pcl3 = np.memmap(ptcl_path3+sys.argv[1]+'.bin', dtype='float32', mode='r').reshape([-1,4])
	pcl4 = np.memmap(ptcl_path4+sys.argv[1]+'.bin', dtype='float32', mode='r').reshape([-1,4])
	pcl5 = np.memmap(ptcl_path5+sys.argv[1]+'.bin', dtype='float32', mode='r').reshape([-1,4])
	pcl6 = np.memmap(ptcl_path6+sys.argv[1]+'.bin', dtype='float32', mode='r').reshape([-1,4]) 
	f = open(oxts_path1+sys.argv[1]+'.txt', 'r')
	oxts1 = [float(x) for x in f.read().split()]
	f.close()
	f = open(oxts_path2+sys.argv[1]+'.txt', 'r')
	oxts2 = [float(x) for x in f.read().split()]
	f.close()
	f = open(oxts_path3+sys.argv[1]+'.txt', 'r')
	oxts3 = [float(x) for x in f.read().split()]
	f.close()
	f = open(oxts_path4+sys.argv[1]+'.txt', 'r')
	oxts4 = [float(x) for x in f.read().split()]
	f.close()
	f = open(oxts_path5+sys.argv[1]+'.txt', 'r')
	oxts5 = [float(x) for x in f.read().split()]
	f.close()
	f = open(oxts_path6+sys.argv[1]+'.txt', 'r')
	oxts6 = [float(x) for x in f.read().split()]
	f.close()
	points_oxts_primary = (pcl1,oxts1)
	points_oxts_secondary = []
	points_oxts_secondary.append((pcl2,oxts2))
	pcl2 = convert(points_oxts_primary, points_oxts_secondary)
	points_oxts_secondary = []
	points_oxts_secondary.append((pcl3,oxts3))
	pcl3 = convert(points_oxts_primary, points_oxts_secondary)
	points_oxts_secondary = []
	points_oxts_secondary.append((pcl4,oxts4))
	pcl4 = convert(points_oxts_primary, points_oxts_secondary)
	points_oxts_secondary = []
	points_oxts_secondary.append((pcl5,oxts5))
	pcl5 = convert(points_oxts_primary, points_oxts_secondary)
	points_oxts_secondary = []
	points_oxts_secondary.append((pcl6,oxts6))
	pcl6 = convert(points_oxts_primary, points_oxts_secondary)	
	pcls.append(pcl1)
	pcls.append(pcl2)
	pcls.append(pcl3)
	pcls.append(pcl4)
	pcls.append(pcl5)
	pcls.append(pcl6)

	pcds = []
	colors = [[0, 0.4, 1], [0, 1, 0.3], [0.0, 0.4, 0.5], [0.1, 0.7, 0.4], [0, 0.8, 0.5], 
			[0.9, 1, 1], [1, 1, 0], [1, 0.5, 0.5], [1, 0, 0.5], [1, 0, 1]]  # blue, green, orange, red

	if len(sys.argv) > 11:
		print('cannot support more than 4 inputs now')
		sys.exit(0)

	for pcl in pcls:
		# pcl = np.fromfile(name, dtype=np.float32, count=-1).reshape([-1,4])
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])
		pcd.paint_uniform_color(colors.pop(0))
		pcds.append(pcd)


	## get bbox

	# pcl_bbox = np.fromfile(sys.argv[2], dtype=np.float32, count=-1).reshape([-1,3])  # [-1,4]
	# pcd_bbox = o3d.geometry.PointCloud()
	# pcd_bbox.points = o3d.utility.Vector3dVector(pcl_bbox[:,:3])
	# pcd_bbox.paint_uniform_color([1, 0.1, 0.1])  # pcd2.paint_uniform_color([0, 0, 1])

	# pcds.append(pcd_bbox)

	o3d.visualization.draw_geometries(pcds)

