# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
import torch
from math import sin, cos


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


def merge_carla(pcd_points, oxts):
	transformed_points = []
	for i in range(len(pcd_points)):
		pcd = pcd_points[i]
		# pcd[:,3] = 1
		pcd = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)
        # pcd_data = np.dot(oxts[i], pcd[:,:4].T).T # pcd[:,:4].dot(trans.T)
		transformed_points.append(np.dot(oxts[i], pcd[:,:4].T).T)
	merged_points = np.vstack(transformed_points)
	return merged_points