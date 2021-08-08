import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import open3d as o3d
import ptcl.pointcloud
import ptcl.pcd_merge
import os.path
from os import path


def convert_encoded_bin_to_pcd(src, dst):
    decoded_pcl = ptcl.pointcloud.dracoDecode(open(src, 'rb').read())
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(decoded_pcl))
    o3d.io.write_point_cloud(dst, o3d_pcd)


def convert_decoded_bin_to_pcd(src, dst):
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    o3d.io.write_point_cloud(dst, o3d_pcd)


def merge_bin_to_pcd(bins, oxtss, dst):
	""" merge point clouds from different coordinates (oxts)
		and save to dst
	Args:
		bins (list): list of pcd bins (or numpy arrays) 
		oxtss (list): list of oxts files
		dst (str): dest to save .pcd file
	"""
	points_oxts_primary = (bins[0], oxtss[0])
	points_oxts_secondary = []
	for idx in range(1, len(bins)):
		points_oxts_secondary.append(bins[idx], oxtss[idx])
	merged_pcl = ptcl.pcd_merge.merge(points_oxts_primary, points_oxts_secondary) # np.array of [n,3]
	# save to dst 
	convert_decoded_bin_to_pcd(merged_pcl, dst)


def create_ref(n, frame_id, folder_name):
    if n > 6:
        return
    basepath = './DeepGTAV-data/object-0227-1/'
    pcd_filename = str(frame_id).zfill(6) + ".bin"
    oxts_filename = str(frame_id).zfill(6) + ".txt" 
    ptcl_paths = [basepath + x for x in ['velodyne_2/', 'alt_perspective/0022786/velodyne_2/', 'alt_perspective/0037122/velodyne_2/',
                'alt_perspective/0191023/velodyne_2/', 'alt_perspective/0399881/velodyne_2/', 'alt_perspective/0735239/velodyne_2/']]
    oxts_paths = [basepath + x for x in ['oxts/', 'alt_perspective/0022786/oxts/', 'alt_perspective/0037122/oxts/', 
                'alt_perspective/0191023/oxts/', 'alt_perspective/0399881/oxts/', 'alt_perspective/0735239/oxts/']]
    pcls = []
    oxtss = [] 
    for i in range(n):
        pcls.append(np.memmap(ptcl_paths[i] + pcd_filename, dtype='float32', mode='r').reshape([-1,4]))
        f = open(oxts_paths[i] + oxts_filename, 'r')
        oxtss.append([float(x) for x in f.read().split()])
        f.close()
    points_oxts_primary = (pcls[0], oxtss[0])
    points_oxts_secondary = []
    for i in range(1, n):
        points_oxts_secondary.append((pcls[i],oxtss[i]))
    pcd_name = folder_name + "/" + str(frame_id).zfill(6) + '_' + str(n) + '.pcd'
    if n == 1:
        convert_decoded_bin_to_pcd(np.memmap(ptcl_paths[0] + pcd_filename, dtype='float32', mode='r').reshape([-1,4])[:,:3], pcd_name)
    else:
        pcl = ptcl.pcd_merge.merge(points_oxts_primary, points_oxts_secondary)
        o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcl[:, :3]))
        o3d.io.write_point_cloud(pcd_name, o3d_pcd)


def convert_dis(node_id, frame_id, folder):
    prefix = "node" + str(node_id) + "_frame" + str(frame_id) + "_"
    print(prefix)
    chunks = []
    for file in os.listdir(folder):
        # print(file)
        if prefix in file:
            chunks.append(file)
    # print(chunks)
    pcds = []
    for chunk in chunks:
        print(chunk)
        pcds.append(ptcl.pointcloud.dracoDecode(open(folder + "/" + chunk, 'rb').read()))
    return np.vstack(pcds)


if __name__ == "__main__":
    # folder_name = "gta_ref_frames"
    # os.system('mkdir ' + folder_name)
    # for n in range(1, 7):
    #     for i in range(80):
    #         create_ref(n, i, folder_name)
    # # exit(0)
    basepath = '/home/shawnzhu/DeepGTAV-data/object-0227-1/'
    oxts_paths = [basepath + x for x in ['oxts/', 'alt_perspective/0022786/oxts/', 'alt_perspective/0037122/oxts/', 
                'alt_perspective/0191023/oxts/', 'alt_perspective/0399881/oxts/', 'alt_perspective/0735239/oxts/']]
    folder = sys.argv[1]
    for frame_id in range(10):
        bins = []
        oxtss = []
        dst = folder + "/merged_frame" + str(frame_id) + ".pcd"
        for node_id in range(1):
            oxts_filename = str(frame_id).zfill(6) + ".txt" 
            f = open(oxts_paths[node_id] + oxts_filename, 'r')
            oxtss.append([float(x) for x in f.read().split()])
            bins.append(convert_dis(node_id, frame_id, folder))
        merge_bin_to_pcd(bins, oxtss, dst)
