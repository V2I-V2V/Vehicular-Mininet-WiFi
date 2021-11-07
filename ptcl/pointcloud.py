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


def read_pointcloud(pointcloud_filename, data_type="GTA"):
    """Read point cloud data into a numpy array

    Args:
        pointcloud_filename (str): point cloud data file name

    Returns:
        numpy array: numpy array consisting of the pcd data points (shape [x, 4])
    """
    if data_type == "GTA":
        pcd_file = open(pointcloud_filename, "rb")
        raw_pcd_data = pcd_file.read()
        pcd_data = np.frombuffer(raw_pcd_data, dtype=np.float32).reshape([-1,4])
    elif data_type == "Carla":
        pcd_data = np.load(pointcloud_filename)
        pcd_data = pcd_data.astype(np.float32)
        tranformation_matrix_prefix, file_extension = os.path.splitext(pointcloud_filename)
        # trans = np.load(tranformation_matrix_prefix + ".trans.npy")
        # pcd_data[:,3] = 1
        # pcd_data = np.dot(trans, pcd_data[:,:4].T).T # pcd[:,:4].dot(trans.T)
    return pcd_data


def read_all_oxts(oxts_dir):
    all_oxts = []
    for i in range(config.MAX_FRAMES):
        oxts_f_name = oxts_dir + "%06d.txt"%i
        all_oxts.append(read_oxts(oxts_f_name))
    return all_oxts


def read_oxts(oxts_filename, data_type="GTA"):
    if data_type == "GTA":
        oxts_file = open(oxts_filename, 'rb')
        oxts_data = oxts_file.read()
    elif data_type == "Carla":
        oxts_data = np.load(oxts_filename).tobytes()
    return oxts_data


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
    
