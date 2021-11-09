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
    qb_range = np.arange(7, 13)
    cl_range = np.arange(1, 10)
    usrname = getpass.getuser()
    datadir = '/home/' + usrname + '/V2I+V2V/carla-bin/lidar/'
    vehicle_dir = os.listdir(datadir)
    qb_to_size_dict = collections.defaultdict(list)
    cl_to_size_dict = collections.defaultdict(list)
    qb_to_detected_space = collections.defaultdict(list)
    t_start = time.time()
    for v_dir in vehicle_dir:
        velodyne_dir = datadir + v_dir + '/velodyne'
        if os.path.isdir(velodyne_dir):
            ptcls = os.listdir(velodyne_dir)
            for ptcl_name in ptcls:
                # print(ptcl_name)
                ptcl = ptcl_utils.read_ptcl_data(velodyne_dir + '/' + ptcl_name)
                raw_size = os.path.getsize(velodyne_dir + '/' + ptcl_name) / 1000
                qb_to_size_dict['raw'].append(raw_size)
                cl_to_size_dict['raw'].append(raw_size)
                # p2, p1, best_model, _ = my_ransac_v5(ptcl, 10000, P=0.8, distance_threshold=0.35,
                #                                      lidar_height=-2.03727 + 0.1, lidar_height_down=-2.03727 - 0.1,
                #                                      use_all_sample=True)
                # ptcl[:, 3] = 0  # object
                # ptcl[p2, 3] = 1  # ground
                # raw_grid, _ = ptcl_utils.calculate_grid_label_ransac(1, ptcl)
                # raw_space = len(raw_grid[raw_grid != 0])
                # qb_to_detected_space['raw'].append(raw_space)
                for qb in qb_range:
                    encoded, ratio = pointcloud.dracoEncode(ptcl, 10, qb)  # make compression level const here
                    decoded = pointcloud.dracoDecode(encoded)
                    decoded = np.hstack([decoded, np.ones((ptcl.shape[0], 1), dtype=np.float32)])
                    encoded_size = raw_size * ratio
                    qb_to_size_dict[qb].append(encoded_size)
                    assert ptcl.shape[0] == decoded.shape[0]

                for cl in cl_range:
                    encoded, ratio = pointcloud.dracoEncode(ptcl, cl, 14)  # make ab const here
                    decoded = pointcloud.dracoDecode(encoded)
                    decoded = np.hstack([decoded, np.ones((ptcl.shape[0], 1), dtype=np.float32)])
                    encoded_size = raw_size * ratio
                    cl_to_size_dict[cl].append(encoded_size)
                    assert ptcl.shape[0] == decoded.shape[0]
                    # p2, p1, best_model, _ = my_ransac_v5(decoded, 10000, P=0.8, distance_threshold=0.35,
                    #                                      lidar_height=-2.03727 + 0.1, lidar_height_down=-2.03727 - 0.1,
                    #                                      use_all_sample=True)
                    # decoded[:, 3] = 0  # object
                    # decoded[p2, 3] = 1  # ground
                    # decoded_grid, _ = ptcl_utils.calculate_grid_label_ransac(1, decoded)
                    # decoded_space = len(decoded_grid[decoded_grid != 0])
                    # qb_to_detected_space[qb].append(decoded_space)

    print('Time taken:', time.time() - t_start)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cnt = 0
    ax.boxplot(qb_to_size_dict['raw'], positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
    cnt += 1
    for qb, sizes in qb_to_size_dict.items():
        if qb != 'raw':
            ax.boxplot(sizes, positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
            cnt += 1

    tick_labels = ['raw']
    for qb in qb_range:
        tick_labels.append(str(qb))
    ax.set_xticks(np.arange(len(qb_to_size_dict.keys())))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('Data Size (KB)')
    ax.set_xlabel('Compression Level (qb)')
    plt.savefig('size_vs_compression_qb.pdf')

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cnt = 0
    for cl, sizes in cl_to_size_dict.items():
        if cl != 'raw':
            ax.boxplot(sizes, positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
            cnt += 1
    # ax.boxplot(cl_to_size_dict['raw'], positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)

    tick_labels = []
    for cl in cl_range:
        tick_labels.append(str(cl))
    # tick_labels.append('raw')
    ax.set_xticks(np.arange(len(cl_to_size_dict.keys())-1))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('Data Size (KB)')
    ax.set_xlabel('Compression Level (cl)')
    plt.savefig('size_vs_compression_cl_14qb.pdf')

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # cnt = 0
    # ax.boxplot(qb_to_detected_space['raw'], positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
    # cnt += 1
    # for qb, sizes in qb_to_detected_space.items():
    #     if qb != 'raw':
    #         ax.boxplot(sizes, positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
    #         cnt += 1
    #
    # tick_labels = ['raw']
    # for qb in qb_range:
    #     tick_labels.append(str(qb))
    # ax.set_xticks(np.arange(len(qb_to_detected_space.keys())))
    # ax.set_xticklabels(tick_labels)
    # ax.set_ylabel('Detected Space (m^2)')
    # ax.set_xlabel('Compression Level (qb)')
    # plt.savefig('space_vs_compression.pdf')


    # ptcl = ptcl_utils.read_ptcl_data('/home/' + usrname + '/V2I+V2V/carla-bin/lidar/86/velodyne/1000.bin')
    # # ptcl = ptcl_utils.read_ptcl_data('000001.bin')
    # cl, qb = 10, 7
    # encoded, ratio = pointcloud.dracoEncode(ptcl, cl, qb)
    # decoded = pointcloud.dracoDecode(encoded)
    # print(1 - ratio)
    # # print(2*ratio)
    # print(ptcl.shape[0], decoded.shape[0])
    # ptcl_utils.draw_3d(decoded)
    # ptcl_utils.save_ptcl(decoded, './1000-%d-%d.bin' % (cl, qb))
