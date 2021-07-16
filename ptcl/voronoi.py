# -*- coding: utf-8 -*-
# Filename : voronoi_edge_adapt
__author__ = 'Xumiao Zhang'

import os
import sys
import time
import numpy as np
from math import sin, cos, sqrt
import time
from scipy.spatial import Voronoi

data_path = '/home/harry/5g/edge/DeepGTAV-data/object-0226/'
save_path = '/home/harry/5g/edge/'
ptcl_path = data_path + 'velodyne_2/'
oxts_path = data_path + 'oxts/'
self_path = data_path + 'ego_object/'
alt_path = data_path + 'alt_perspective/'

def calculate_pb(p1, p2, normalize=True):
	x1, y1 = p1
	x2, y2 = p2

	if x1 == x2 and y1 == y2:  # three points in line
		print('Two same coordinates!')
		return None
	elif x1 == x2:
		a = 0
		b = 1
	elif y1 == y2:
		a = 1
		b = 0
	else:
		a = x2 - x1
		b = y2 - y1

	if normalize:
		r = np.sqrt(a ** 2 + b ** 2)
		a = a / r
		b = b / r
	c = -(x1+x2)/2*a - (y1+y2)/2*b
	
	if x1*a + y1*b + c < 0:
		a, b, c = -a, -b, -c
	# print(a,b,c)
	return a, b, c  # ax+by+c=0, ax+by+c_self=0, ax+by+c_other=0

def calculate_perpendicular_power(p1, p2, normalize=True, r1=1, r2=1):
	# print('\n', p1, p2)
	x1, y1 = p1
	x2, y2 = p2

	if x1 == x2 and y1 == y2:  # three points in line
		print('Two same coordinates!')
		return None
	# elif x1 == x2:
	# 	a = 0
	# 	b = 1
	# elif y1 == y2:
	# 	a = 1
	# 	b = 0
	else:
		a = x2 - x1
		b = y2 - y1
	if normalize:
		r = np.sqrt(a ** 2 + b ** 2)
		a = a / r
		b = b / r
	# print(r1,r2)
	c = ((x1**2+y1**2-r1**2)-(x2**2+y2**2-r2**2))/(2*r)
	# c = -(x1+(x2-x1)*ratio/(ratio+1))*a - (y1+(y2-y1)*ratio/(ratio+1))*b
	# c = -(x1+x2)*ratio/(ratio+1)*a - (y1+y2)*ratio/(ratio+1)*b
	
	if (x1*a + y1*b + c) * (x2*a + y2*b + c) < 0:
		if (x1*a + y1*b + c) < 0:
			a, b, c = -a, -b, -c
	else:
		if (x1*a + y1*b + c) > 0 and r1 < r2:
			a, b, c = -a, -b, -c
		if (x1*a + y1*b + c) < 0 and r1 > r2:
			a, b, c = -a, -b, -c
	# print(ratio)
	# print(a,b,c)
	return a, b, c  # ax+by+c=0

def calculate_perpendicular_power2(p1, p2, normalize=True, r1=1, r2=1, add=0.3, minu=0.3):
	# print('\n', p1, p2)
	x1, y1 = p1
	x2, y2 = p2

	if x1 == x2 and y1 == y2:  # three points in line
		print('Two same coordinates!')
		return None
	# elif x1 == x2:
	# 	a = 0
	# 	b = 1
	# elif y1 == y2:
	# 	a = 1
	# 	b = 0
	else:
		a = x2 - x1
		b = y2 - y1

	if normalize:
		r = np.sqrt(a ** 2 + b ** 2)
		a = a / r
		b = b / r
	# print(r1,r2)
	c = ((x1**2+y1**2-r1**2)-(x2**2+y2**2-r2**2))/(2*r)
	c1 = ((x1**2+y1**2-(r1*(1-minu))**2)-(x2**2+y2**2-(r2*(1+add))**2))/(2*r)
	c2 = ((x1**2+y1**2-(r1*(1+add))**2)-(x2**2+y2**2-(r2*(1-minu))**2))/(2*r)
	
	if (x1*a + y1*b + c) * (x2*a + y2*b + c) < 0:
		if (x1*a + y1*b + c) < 0:
			a, b, c, c1, c2 = -a, -b, -c, -c1, -c2
	else:
		if (x1*a + y1*b + c) > 0 and r1 < r2:
			a, b, c, c1, c2 = -a, -b, -c, -c1, -c2
		if (x1*a + y1*b + c) < 0 and r1 > r2:
			a, b, c, c1, c2 = -a, -b, -c, -c1, -c2
	# print(ratio)
	# print(a,b,c)
	return a, b, c, c1, c2  # ax+by+c=0

def transformation(oxts1, oxts2):
	### transformation matrix - translation (to the perspective of oxts1)
	da = oxts2[0] - oxts1[0]  # south --> north
	db = oxts2[1] - oxts1[1]  # east --> west
	dx = da * cos(oxts1[5]) + db * sin(oxts1[5])
	dy = da * -sin(oxts1[5]) + db * cos(oxts1[5])
	translation = [dx, dy]
	# print("Translation: ", dx ,dy)
	return translation

def voronoi(oxtsSet):
	indexSet = []
	vehSet = []

	for idx, oxts in enumerate(oxtsSet):
		if oxts:
			vehSet.append(oxts)
			indexSet.append(idx)
	num_vehs = len(indexSet)

	pbSet = np.asarray([None] * len(oxtsSet))
	neighbors = [[] for _ in range(len(oxtsSet))]
	pbSet = [[] for _ in range(num_vehs)]
	
	if num_vehs == 1:
		return pbSet, None, indexSet
	elif num_vehs == 2:
		neighbors[0].append(1)
		neighbors[1].append(0)
	else:
		voi = Voronoi(np.array(vehSet)[:,:2])
		ridge_points = voi.ridge_points
		for pair in ridge_points:
			a, b = pair
			neighbors[a].append(b)
			neighbors[b].append(a)

	vehs = [[] for _ in range(num_vehs)]
	for i in range(num_vehs):
		for j in neighbors[i]:
			vehs[i].append(transformation(vehSet[i], vehSet[j]))  # relative position
		for x, j in enumerate(neighbors[i]):
			pbSet[i].append(calculate_pb([0,0], vehs[i][x]))  # current car: [0,0]
	pbSet = [np.asarray(ps) for ps in pbSet]
	return pbSet, neighbors, indexSet

def voronoi_bw(oxtsSet, bwSet=None):
	if bwSet is None:
		bwSet = np.ones(len(oxtsSet))
	
	indexSet = []
	vehSet = []

	for idx, oxts in enumerate(oxtsSet):
		if oxts:
			vehSet.append(oxts)
			indexSet.append(idx)
	num_vehs = len(indexSet)

	pbSet = np.asarray([None] * len(oxtsSet))
	neighbors = [[] for _ in range(len(oxtsSet))]
	pbSet = [[] for _ in range(num_vehs)]
	
	if num_vehs == 1:
		return pbSet, None, indexSet
	elif num_vehs == 2:
		neighbors[0].append(1)
		neighbors[1].append(0)
	else:
		voi = Voronoi(np.array(vehSet)[:,:2])
		ridge_points = voi.ridge_points
		for pair in ridge_points:
			a, b = pair
			neighbors[a].append(b)
			neighbors[b].append(a)

	vehs = [[] for _ in range(num_vehs)]
	for i in range(num_vehs):
		for j in neighbors[i]:
			vehs[i].append(transformation(vehSet[i], vehSet[j]))  # relative position
		for x, j in enumerate(neighbors[i]):
			if bwSet[indexSet[i]] > 1000 or bwSet[indexSet[j]] > 1000:
				pbSet[i].append(calculate_perpendicular_power([0,0], vehs[i][x], r1=bwSet[indexSet[i]]/100, r2=bwSet[indexSet[j]]/100))  # current car: [0,0]
			elif bwSet[indexSet[i]] > 100 or bwSet[indexSet[j]] > 100:
				pbSet[i].append(calculate_perpendicular_power([0,0], vehs[i][x], r1=bwSet[indexSet[i]]/10, r2=bwSet[indexSet[j]]/10))  # current car: [0,0]
			else:
				pbSet[i].append(calculate_perpendicular_power([0,0], vehs[i][x], r1=bwSet[indexSet[i]]/4, r2=bwSet[indexSet[j]]/4))  # current car: [0,0]
	pbSet = [np.asarray(ps) for ps in pbSet]
	return pbSet, neighbors, indexSet

def voronoi_adapt(oxtsSet, bwSet=None):
	if bwSet is None:
		bwSet = np.ones(len(oxtsSet))
	
	indexSet = []
	vehSet = []

	for idx, oxts in enumerate(oxtsSet):
		if oxts:
			vehSet.append(oxts)
			indexSet.append(idx)
	num_vehs = len(indexSet)

	pbSet = np.asarray([None] * len(oxtsSet))
	neighbors = [[] for _ in range(len(oxtsSet))]
	pbSet = [[] for _ in range(num_vehs)]
	
	if num_vehs == 1:
		return pbSet, None, indexSet
	elif num_vehs == 2:
		neighbors[0].append(1)
		neighbors[1].append(0)
	else:
		voi = Voronoi(np.array(vehSet)[:,:2])
		ridge_points = voi.ridge_points
		for pair in ridge_points:
			a, b = pair
			neighbors[a].append(b)
			neighbors[b].append(a)

	vehs = [[] for _ in range(num_vehs)]
	for i in range(num_vehs):
		for j in neighbors[i]:
			vehs[i].append(transformation(vehSet[i], vehSet[j]))  # relative position
		for x, j in enumerate(neighbors[i]):
			if bwSet[indexSet[i]] > 1000 or bwSet[indexSet[j]] > 1000:
				pbSet[i].append(calculate_perpendicular_power2([0,0], vehs[i][x], r1=bwSet[indexSet[i]]/100, r2=bwSet[indexSet[j]]/100))  # current car: [0,0]
			elif bwSet[indexSet[i]] > 100 or bwSet[indexSet[j]] > 100:
				pbSet[i].append(calculate_perpendicular_power2([0,0], vehs[i][x], r1=bwSet[indexSet[i]]/10, r2=bwSet[indexSet[j]]/10))  # current car: [0,0]
			else:
				pbSet[i].append(calculate_perpendicular_power2([0,0], vehs[i][x], r1=bwSet[indexSet[i]]/4, r2=bwSet[indexSet[j]]/4))  # current car: [0,0]
	pbSet = [np.asarray(ps) for ps in pbSet]
	return pbSet, neighbors, indexSet

def voronoi_mask(pcl, ps):
	if len(ps) == 0:
		return pcl
	### perpendicular - self
	xy = pcl[:,:2]
	p_bisectors = ps[:,]
	signs = []
	for p_bisector in p_bisectors:
		tmp = np.matmul(xy, p_bisector[:2])
		signs.append(tmp >= -p_bisector[2])
	filt = np.asarray(signs).sum(axis=0) == len(ps)
	pcl_new = pcl[filt,:]
	return pcl_new

def voronoi_mask_bw(pcl, ps):
	if len(ps) == 0:
		return pcl
	### perpendicular - self
	xy = pcl[:,:2]
	p_bisectors = ps[:,]
	signs = []
	for p_bisector in p_bisectors:
		tmp = np.matmul(xy, p_bisector[:2])
		signs.append(tmp >= -p_bisector[2])
	filt = np.asarray(signs).sum(axis=0) == len(ps)
	pcl_new = pcl[filt,:]
	return pcl_new

def voronoi_mask_adapt(pcl, ps):
	if len(ps) == 0:
		return pcl
	### perpendicular - self
	xy = pcl[:,:2]
	p_selfs = ps[:,(0,1,3)]
	signs = []
	for p_self in p_selfs:
		tmp = np.matmul(xy, p_self[:2])
		signs.append(tmp >= -p_self[2])
	filt = np.asarray(signs).sum(axis=0) == len(ps)
	pcl_1 = pcl[filt,:]
	pcl_234 = pcl[~filt,:]
	### perpendicular - bisector
	xy = pcl_234[:,:2]
	p_bisectors = ps[:,(0,1,2)]
	signs = []
	for p_bisector2 in p_bisectors:
		tmp = np.matmul(xy, p_bisector2[:2])
		signs.append(tmp >= -p_bisector2[2])
	filt = np.asarray(signs).sum(axis=0) == len(ps)
	pcl_2 = pcl_234[filt,:]
	pcl_34 = pcl_234[~filt,:]
	### perpendicular - others
	xy = pcl_34[:,:2]
	p_others = ps[:,(0,1,4)]
	signs = []
	for p_other in p_others:
		tmp = np.matmul(xy, p_other[:2])
		signs.append(tmp >= -p_other[2])
	filt = np.asarray(signs).sum(axis=0) == len(ps)
	pcl_3 = pcl_34[filt,:]
	pcl_4 = pcl_34[~filt,:]
	# pcl_new = np.vstack((pcl_1, pcl_2))
	return pcl_1, pcl_2, pcl_3, pcl_4


if __name__ == '__main__':	
	pclSet = []
	oxtsSet = []
	vehs = [[0,0]]
	
	pcl0 = np.memmap("../resources/sample_data_for_merging/ego/velodyne/000003.bin", dtype='float32', mode='r').reshape([-1,4])
	with open("../resources/sample_data_for_merging/ego/oxts/000003.txt", 'r') as f:
		oxts0 = [float(x) for x in f.read().split()]
	pcl1 = np.memmap("../resources/sample_data_for_merging/leftturn/velodyne/000003.bin", dtype='float32', mode='r').reshape([-1,4])
	with open("../resources/sample_data_for_merging/leftturn/oxts/000003.txt", 'r') as f:
		oxts1 = [float(x) for x in f.read().split()]
	pcl2 = np.memmap("../resources/sample_data_for_merging/straight/velodyne/000003.bin", dtype='float32', mode='r').reshape([-1,4])
	with open("../resources/sample_data_for_merging/straight/oxts/000003.txt", 'r') as f:
		oxts2 = [float(x) for x in f.read().split()]

	oxtsSet.append(oxts0)
	oxtsSet.append(oxts1)
	oxtsSet.append(oxts2)
	pclSet.append(pcl0)
	pclSet.append(pcl1)
	pclSet.append(pcl2)


	### Voronoi
	print("Voronoi ...")
	pbSet, neighbors, indexSet = voronoi(oxtsSet)
	for pb in pbSet:
		print(pb)
	print(neighbors)
	print(indexSet)
	for i, pcl in enumerate(pclSet):
		pcl_new = voronoi_mask(pcl, pbSet[i])
		print(f"Full: {pclSet[i].shape[0]}; Part: {pcl_new.shape[0]}")
	print()


	### Voronoi_BW
	print("Voronoi BW ...")
	bwSet = [4, 6, 8]
	pbSet, neighbors, indexSet = voronoi_bw(oxtsSet, bwSet)
	for pb in pbSet:
		print(pb)
	print(neighbors)
	print(indexSet)
	pcl_news = []
	for i, pcl in enumerate(pclSet):
		pcl_new = voronoi_mask_bw(pcl, pbSet[i])
		print(f"Full: {pclSet[i].shape[0]}; Part: {pcl_new.shape[0]}")
	print()


	### Voronoi_adapt
	print("Voronoi Adapt...")
	pbSet, neighbors, indexSet = voronoi_adapt(oxtsSet, bwSet)
	for pb in pbSet:
		print(pb)
	print(neighbors)
	print(indexSet)
	pcl_news = []
	for i, pcl in enumerate(pclSet):
		pcl_1, pcl_2, pcl_3, pcl_4 = voronoi_mask_adapt(pcl, pbSet[i])
		print(f"Full: {pclSet[i].shape[0]}; Part: {pcl_1.shape[0]}, {pcl_2.shape[0]}, {pcl_3.shape[0]}, {pcl_4.shape[0]}")