import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import open3d as o3d
import ptcl.pointcloud
import ptcl.pcd_merge
import os.path
from os import path
from run_experiment import parse_config_from_file
from analysis.util import get_stats_on_one_run
import config
import getpass


def convert_encoded_bin_to_pcd(src, dst):
    decoded_pcl = ptcl.pointcloud.dracoDecode(open(src, 'rb').read())
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(decoded_pcl))
    o3d.io.write_point_cloud(dst, o3d_pcd)


def convert_decoded_bin_to_pcd(src, dst):
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src[:, :3]))
    o3d.io.write_point_cloud(dst, o3d_pcd)


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
        convert_decoded_bin_to_pcd(bins[0], dst)
        return
    else:
        extended_bin = np.concatenate((bins[0], np.ones((bins[0].shape[0], 1),dtype=np.float32)), axis=1)
        points_oxts_primary = (extended_bin, oxts[0])
    points_oxts_secondary = []
    for idx in range(1, len(bins)):
        extended_bin = np.concatenate((bins[idx], np.ones((bins[idx].shape[0], 1),dtype=np.float32)), axis=1)
        points_oxts_secondary.append((extended_bin, oxts[idx]))
    print(dst)
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


def get_dis(node_id, frame_id, folder):
    prefix = "node" + str(node_id) + "_frame" + str(frame_id) + "_"
    # print("prefix",prefix)
    # print("file", folder)
    chunks = []
    for file in os.listdir(folder):
        # print(file)
        if prefix in file:
            # print("prefix", prefix)
            # print("file", file)
            chunks.append(file)
    # print(chunks)
    pcds = []
    for chunk in chunks:
        # print(chunk)
        pcds.append(ptcl.pointcloud.dracoDecode(open(folder + "/"  + chunk, 'rb').read()))
    if pcds == []:
        return None
    else:
        return np.vstack(pcds)


if __name__ == "__main__":
    process_type = sys.argv[1]
    data_folder = sys.argv[2]
    if process_type == "ref":
        # data_folder = "gta_ref_frames"
        os.system('mkdir ' + data_folder)
        for n in range(1, 7):
            for i in range(80):
                create_ref(n, i, data_folder)
    elif process_type == "dis":
        basepath = '/home/'+ getpass.getuser() + '/DeepGTAV-data/object-0227-1/'
        oxts_paths = [basepath + x for x in ['oxts/', 'alt_perspective/0022786/oxts/', 'alt_perspective/0037122/oxts/', 
                    'alt_perspective/0191023/oxts/', 'alt_perspective/0399881/oxts/', 'alt_perspective/0735239/oxts/']]
        
        pcd_folder = data_folder + "/output"
        configs = parse_config_from_file(data_folder + '/config.txt')
        num_of_nodes = int(configs["num_of_nodes"])
        node_to_latency, node_to_encode_choices = get_stats_on_one_run(data_folder, num_of_nodes, configs["helpee_conf"])
        # print(node_to_latency)
        # print(len(node_to_encode_choices[0]))
        node_ids = list(node_to_encode_choices.keys())
        print(node_to_encode_choices)
        max_frame_id = 0
        for node_id in node_ids:
            for k, v in node_to_encode_choices[node_id].items():
                if k > max_frame_id:
                    max_frame_id = k
        print(max_frame_id)
        for frame_id in range(0, max_frame_id + 1): # max_frame_id, max_frame_id + 1
            bins = []
            oxtss = []
            dst = pcd_folder + "/merged_frame" + str(frame_id) + ".pcd"
            for node_id in node_ids:
                if frame_id not in node_to_encode_choices[node_id]:
                    print("id skipped")
                    continue
                oxts_filename = str(frame_id % config.MAX_FRAMES).zfill(6) + ".txt" 
                f = open(oxts_paths[node_id] + oxts_filename, 'r')
                oxtss.append([float(x) for x in f.read().split()])
                dis = get_dis(node_id, frame_id, pcd_folder)
                if dis is not None:
                    bins.append(dis)
            merge_bin_to_pcd(bins, oxtss, dst)
