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


def merge_bin_to_pcd(bins, oxts, dst):




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


def convert_dis(node_id, frame_id):
    prefix = "node" + node_id + "_frame" + str(frame_id) + "_"
    chunks = []
    for file in os.listdir(sys.argv[1]):
        # print(file)
        if prefix in file:
            chunks.append(file)
    # print(chunks)
    pcds = []
    for chunk in chunks:
        print(chunk)
        pcds.append(ptcl.pointcloud.dracoDecode(open(sys.argv[1] + "/" + chunk, 'rb').read()))
    merged_from_chunks = np.vstack(pcds)
    # merge among nodes


    convert_decoded_bin_to_pcd(merged_from_chunks, sys.argv[1] + "/node0_frame" + str(i) + ".pcd")


if __name__ == "__main__":
    folder_name = "gta_ref_frames"
    os.system('mkdir ' + folder_name)
    for n in range(1, 7):
        for i in range(80):
            create_ref(n, i, folder_name)
    # exit(0)

    for i in range(50):
        


def main():
    decoded_pcl = pointcloud.dracoDecode(open('node0_0.bin', 'rb').read())
    decoded_pcl = np.concatenate((decoded_pcl[:, :3], np.zeros((decoded_pcl.shape[0],1), dtype=np.float32)), axis=1)
    with open('node.bin', 'wb') as f:
        f.write(decoded_pcl)
        f.close()
#     exit(0)
    # for i in range(700):
    #     # bin2pcd(np.fromfile(str(i).zfill(6) + ".bin", dtype=np.float32).reshape((-1, 4))[:, 0:3], str(i).zfill(6) + ".pcd")
    #     if path.exists('node0_' + str(i) + '.bin'):
    #         decoded_pcl = pointcloud.dracoDecode(open('node0_' + str(i) + '.bin', 'rb').read())
    #         # Convert to Open3D point cloud
    #         o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(decoded_pcl))
    #         # Save to whatever format you like
    #         o3d.io.write_point_cloud('node0_' + str(i) + ".pcd", o3d_pcd)
    # decoded_pcl = pointcloud.dracoDecode(open('node0_0.bin', 'rb').read())
    # # Convert to Open3D point cloud
    # o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(decoded_pcl))
    # # Save to whatever format you like
    # o3d.io.write_point_cloud('node0_0.pcd', o3d_pcd)

