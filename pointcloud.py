# Read and process point cloud
import numpy as np
import config

def read_all_pointclouds(pointcloud_dir):
    all_frames = []
    for i in range(config.MAX_FRAMES):
        pcd_f_name = pointcloud_dir + "%06d.bin"%i
        all_frames.append(read_pointcloud(pcd_f_name))
    return all_frames


def read_pointcloud(pointcloud_filename):
    """Read point cloud data into a numpy array

    Args:
        pointcloud_filename (str): point cloud data file name

    Returns:
        byte array: byte array consisting of the pcd data points
    """
    pcd_file = open(pointcloud_filename, "rb")
    pcd_data = pcd_file.read()
    return pcd_data

def read_all_oxts(oxts_dir):
    all_oxts = []
    for i in range(config.MAX_FRAMES):
        oxts_f_name = oxts_dir + "%06d.txt"%i
        all_oxts.append(read_oxts(oxts_f_name))
    return all_oxts

def read_oxts(oxts_filename):
    oxts_file = open(oxts_filename, 'rb')
    oxts_data = oxts_file.read()
    return oxts_data

def compress_pointcloud(compression_level, pointcloud):
    # TODO
    pass
