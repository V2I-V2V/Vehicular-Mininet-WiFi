import numpy as np
import sys
import time
import torch
from math import sin, cos
import getpass
import TrakoDracoPy

def dracoEncode(points, cl, qb):
    if points.shape[0] == 0:
        points = np.zeros((1, 4))
        print("empty data to encode!")
        # return b'', 0
    encode_buf = TrakoDracoPy.encode_point_cloud_to_buffer(points[:,:3].flatten(), position=True, 
        sequential=False, remove_duplicates=False, quantization_bits=qb, compression_level=cl,
        quantization_range=-1, quantization_origin=None, create_metadata=False)
    ratio = len(encode_buf) / (12.0 * points.shape[0])
    return encode_buf, ratio


def dracoDecode(pc_encoded):
    if len(pc_encoded) == 0:
        return np.empty((0,3))
    decode_buf = TrakoDracoPy.decode_point_cloud_buffer(pc_encoded)
    pc = np.asarray(decode_buf.points).astype(np.float32).reshape([-1,3])
    return pc


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


def merge_bin_to_pcd(bins, oxts, dst):
    """ merge point clouds from different coordinates (oxts)
        and save to dst
    Args:
        bins (list): list of pcd bins (or numpy arrays) 
        oxts (list): list of oxts files
        dst (str): dest to save .pcd file
    """	
    if len(bins) == 0:
        return
    elif len(bins) == 1:
        extended_bin = np.concatenate((bins[0], np.ones((bins[0].shape[0], 1),dtype=np.float32)), axis=1)
        with open(dst, 'w') as f:
            extended_bin.tofile(f)
        return
    else:
        extended_bin = np.concatenate((bins[0], np.ones((bins[0].shape[0], 1),dtype=np.float32)), axis=1)
        points_oxts_primary = (extended_bin, oxts[0])
    points_oxts_secondary = []
    for idx in range(1, len(bins)):
        extended_bin = np.concatenate((bins[idx], np.ones((bins[idx].shape[0], 1),dtype=np.float32)), axis=1)
        points_oxts_secondary.append((extended_bin, oxts[idx]))
    print(dst)
    merged_pcl = merge(points_oxts_primary, points_oxts_secondary) # np.array of [n,4]
    with open(dst, 'w') as f:
        merged_pcl.tofile(f)

import itertools
 

def findsubsets(s, n):
    return list(itertools.combinations(s, n))


def convert_tuple_to_string(tup, frame_id):
    s = "%06d_"%frame_id
    for item in tup:
        s += str(item)
    s += '.bin'
    return s


def merge_carla(pcd_points, oxts):
    transformed_points = []
    for i in range(len(pcd_points)):
        pcd = pcd_points[i]
        pcd = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)
    # pcd_data = np.dot(oxts[i], pcd[:,:4].T).T # pcd[:,:4].dot(trans.T)
    transformed_points.append(np.dot(oxts[i], pcd[:,:4].T).T)
    merged_points = np.vstack(transformed_points)



if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'gta':
        basepath = '/home/ryanzhu/DeepGTAV-Data/object-0227-1/'
        pcd_paths = [basepath + x for x in ['velodyne_2/', 'alt_perspective/0022786/velodyne_2/', 'alt_perspective/0037122/velodyne_2/', 
                        'alt_perspective/0191023/velodyne_2/', 'alt_perspective/0399881/velodyne_2/', 'alt_perspective/0735239/velodyne_2/']]
        oxts_paths = [basepath + x for x in ['oxts/', 'alt_perspective/0022786/oxts/', 'alt_perspective/0037122/oxts/', 
                        'alt_perspective/0191023/oxts/', 'alt_perspective/0399881/oxts/', 'alt_perspective/0735239/oxts/']]
        pcd_folder = '/home/'+ getpass.getuser() + "/all-output/"
    elif mode == 'carla':
        basepath = '/home/ryanzhu/mvp/data/single-vehicle/lidar/'
        pcd_paths = [basepath + x for x in ['97/', '86/', '108/', 
                        '130/', '152/', '174/', '119', '141', '163', '185']]
        oxts_paths = [basepath + x for x in ['97/', '86/', '108/', 
                        '130/', '152/', '174/', '119', '141', '163', '185']]
        pcd_folder = '/home/'+ getpass.getuser() + "/carla-all-output/"
    s = {0, 1, 2, 3, 4, 5}
    # find all subsets
    for n in range(1, 7):
        subset = findsubsets(s, n)
        for set in subset:
            for frame_id in range(0, 80):
                bins = []
                oxtss = []
                if mode == 'carla':
                    for node_id in set:
                        pcd_filename = "%d.npy"%(frame_id+1000)
                        oxts_filename = "%d.trans.npy"%(frame_id+1000)
                        pcl = np.load(pcd_paths[node_id] + pcd_filename)
                        oxts = np.load(oxts_paths[node_id] + oxts_filename)
                        encoded, _ = dracoEncode(pcl, 10, 12)
                        decoded = dracoDecode(encoded)
                        bins.append(pcl)
                        oxtss.append(oxts)
                    merged_pcl = merge_carla(bins, oxtss)
                    dst = pcd_folder + convert_tuple_to_string(set, frame_id)
                    with open(dst, 'w') as f:
                        merged_pcl.tofile(f)
                elif mode == 'gta':
                    for node_id in set:
                        pcd_filename = "%06d.bin"%frame_id
                        oxts_filename = "%06d.txt"%frame_id 
                        f = open(oxts_paths[node_id] + oxts_filename, 'r')
                        pcl = np.memmap(pcd_paths[node_id] + pcd_filename, dtype='float32', mode='r').reshape([-1,4])
                        encoded, _ = dracoEncode(pcl, 10, 12)
                        decoded = dracoDecode(encoded)
                        bins.append(decoded)
                        oxtss.append([float(x) for x in f.read().split()])
                    dst = pcd_folder + convert_tuple_to_string(set, frame_id)
                    merge_bin_to_pcd(bins, oxtss, dst)



