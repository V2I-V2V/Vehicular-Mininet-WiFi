# Read and process point cloud
import os, sys
import numpy as np
import config
import TrakoDracoPy
import struct

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


def dracoEncode(points, cl, qb):
    encode_buf = TrakoDracoPy.encode_point_cloud_to_buffer(points[:,:3].flatten(), position=True, 
        sequential=False, remove_duplicates=False, quantization_bits=qb, compression_level=cl,
        quantization_range=-1, quantization_origin=None, create_metadata=False)
    ratio = len(encode_buf) / (12.0 * points.shape[0])
    return encode_buf, ratio

def dracoDecode(pc_encoded):
    decode_buf = TrakoDracoPy.decode_point_cloud_buffer(pc_encoded)
    pc = np.asarray(decode_buf.points).astype(np.float32).reshape([-1,3])
    return pc
    

def compress_pointcloud(compression_level, pointcloud):
    # TODO
    pass
