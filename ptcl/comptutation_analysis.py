import sys, os
import time

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointcloud
import ptcl_utils
import getpass
import collections
import matplotlib.pyplot as plt
import matplotlib
from ground import my_ransac_v5

matplotlib.rc('font', **{'size': 18})

if __name__ == '__main__':
    qb_range = np.arange(11, 12)
    usrname = getpass.getuser()
    datadir = '/home/' + usrname + '/Carla/lidar/'
    vehicle_dir = os.listdir(datadir)
    qb_to_size_dict = collections.defaultdict(list)
    qb_to_detected_space = collections.defaultdict(list)
    total_time, total_frame = 0, 0
    for v_dir in vehicle_dir:
        velodyne_dir = datadir + v_dir
        if os.path.isdir(velodyne_dir):
            ptcls = os.listdir(velodyne_dir)
            for ptcl_name in ptcls:
                # print(ptcl_name)
                if 'trans' not in ptcl_name and '.npy' in ptcl_name:
                    ptcl = ptcl_utils.read_ptcl_data(velodyne_dir + '/' + ptcl_name)
                    for qb in qb_range:
                        t_start = time.time()
                        encoded, ratio = pointcloud.dracoEncode(ptcl, 10, qb)  # make compression level const here
                        decoded = pointcloud.dracoDecode(encoded)
                        total_time += (time.time() - t_start)
                        total_frame += 1

    print(total_time/total_frame)

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # cnt = 0
    # ax.boxplot(qb_to_size_dict['raw'], positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
    # cnt += 1
    # for qb, sizes in qb_to_size_dict.items():
    #     if qb != 'raw':
    #         ax.boxplot(sizes, positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
    #         cnt += 1

    # tick_labels = ['raw']
    # for qb in qb_range:
    #     tick_labels.append(str(qb))
    # ax.set_xticks(np.arange(len(qb_to_size_dict.keys())))
    # ax.set_xticklabels(tick_labels)
    # ax.set_ylabel('Data Size (KB)')
    # ax.set_xlabel('Compression Level (qb)')
    # plt.savefig('size_vs_compression.pdf')

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # cnt = 0
    # ax.boxplot(qb_to_detected_space['raw'], positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
    # cnt += 1
    # for qb, sizes in qb_to_detected_space.items():
    #     if qb != 'raw':
    #         ax.boxplot(sizes, positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
    #         cnt += 1

    # tick_labels = ['raw']
    # for qb in qb_range:
    #     tick_labels.append(str(qb))
    # ax.set_xticks(np.arange(len(qb_to_detected_space.keys())))
    # ax.set_xticklabels(tick_labels)
    # ax.set_ylabel('Detected Space (m^2)')
    # ax.set_xlabel('Compression Level (qb)')
    # plt.savefig('space_vs_compression.pdf')
