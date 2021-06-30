# -*- coding: utf-8 -*-
# Filename : bin_merge_gta
__author__ = 'Xumiao Zhang'


# todo: include vehicle height

import numpy as np
import sys
import time
import torch
from math import sin, cos
import open3d as o3d

ptcl_path1 = '/Users/ry4nzzz/Documents/research/V2I+V2V/object-multi-range100-time90/velodyne_2/'
oxts_path1 = '/Users/ry4nzzz/Documents/research/V2I+V2V/object-multi-range100-time90/oxts/'
ptcl_path2 = '/Users/ry4nzzz/Documents/research/V2I+V2V/object-multi-range100-time90/alt_perspective/0005378/velodyne/'
oxts_path2 = '/Users/ry4nzzz/Documents/research/V2I+V2V/object-multi-range100-time90/alt_perspective/0005378/oxts/'

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

def merge(points_oxts_primary, points_oxts_secondary):
	points_0, oxts_0 = points_oxts_primary  # primary vehicle
	# pcl = points_0
	pcl = torch.tensor(points_0)  # torch.from_numpy(points_0)
	for (points, oxts) in points_oxts_secondary:
		points = torch.tensor(points)  # torch.from_numpy(points)
		rotation = rotate(oxts_0, oxts)
		translation = translate(oxts_0, oxts).repeat(np.shape(points)[0],1)  # .numpy()
		# pcl = np.float32(np.concatenate((pcl, np.matmul(points, np.transpose(rotation.numpy())) + translation.numpy())))
		pcl = torch.cat((pcl, torch.mm(points, torch.t(rotation)) + translation))
	return pcl.numpy()


if __name__ == '__main__':
    t = 0
    T = 0
    for i in range(1):
        t0 = time.time()    

    pcl1 = np.memmap(ptcl_path1+sys.argv[1]+'.bin', dtype='float32', mode='r').reshape([-1,4])
    pcl2 = np.memmap(ptcl_path2+sys.argv[1]+'.bin', dtype='float32', mode='r').reshape([-1,4])
    f = open(oxts_path1+sys.argv[1]+'.txt', 'r')
    oxts1 = [float(x) for x in f.read().split()]
    f.close()
    f = open(oxts_path2+sys.argv[1]+'.txt', 'r')
    oxts2 = [float(x) for x in f.read().split()]
    f.close()

    points_oxts_primary = (pcl1,oxts1)
    points_oxts_secondary = []
    points_oxts_secondary.append((pcl2,oxts2))
    t1 = time.time()
    pcl = merge(points_oxts_primary, points_oxts_secondary)
    t2 = time.time()
    # ### save
    # save_filename = save_path + sys.argv[1] + '_' + sys.argv[2]  ## + '_' + sys.argv[3]
    # with open(save_filename + '.bin', 'w') as f:
    # 	pcl.tofile(f)



    t = t + t2-t1
    T = T + t2-t0
