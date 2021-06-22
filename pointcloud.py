# Read and process point cloud
import numpy as np

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


def read_oxts(oxts_filename):
    oxts_file = open(oxts_filename, 'rb')
    oxts_data = oxts_file.read()
    return oxts_data

def compress_pointcloud(compression_level, pointcloud):
    # TODO
    pass
