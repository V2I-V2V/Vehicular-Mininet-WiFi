import os
import sys
# import ptcl.ground
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ptcl.pointcloud import dracoEncode, dracoDecode
import ptcl.ptcl_utils
import getpass
import numpy as np
import multiprocessing
import time

DATASET_DIR = '/home/' + getpass.getuser() + '/Carla/lidar/'

vehicle_id_to_dir = [86, 130, 174, 108, 119, 141, 152, 163, 185, 97]


def get_detected_space(points, detected_spaces, center=(0, 0)):
    merged_pred_grid = \
        ptcl.ptcl_utils.calculate_grid_label_ransac_new(1, points, center=center)
    detected_spaces.append(len(merged_pred_grid[merged_pred_grid != 0]))


def calculate_merged_detection_spaces(v_ids, frame_id, qb_dict, detected_spaces_list):
    ptcls, vehicle_pos = [], {}
    for v_id in v_ids:
        ptcl_name = DATASET_DIR + str(vehicle_id_to_dir[int(v_id) % len(vehicle_id_to_dir)]) \
                    + '/' + str(1000 + frame_id)
        pointcloud = ptcl.ptcl_utils.read_ptcl_data(ptcl_name + '.npy')
        # ptcl.ptcl_utils.calculate_grid_shapely(1, pointcloud)

        trans = np.load(ptcl_name + '.trans.npy')
        encoded, _ = dracoEncode(pointcloud, 10, qb_dict[v_id])
        decoded = dracoDecode(encoded)
        pointcloud = np.concatenate([decoded, np.ones((decoded.shape[0], 1))], axis=1)
        pointcloud = np.dot(trans, pointcloud[:, :4].T).T
        ptcls.append(pointcloud)
        dummy = np.zeros((4, 1))
        dummy[3] = 1
        new_pos = np.dot(trans, dummy).T
        vehicle_pos[int(v_id)] = (new_pos[0, 0], new_pos[0, 1])
    t_s = time.time()
    merged = np.vstack(ptcls)
    merged_pred = ptcl.ptcl_utils.ransac_predict(merged, threshold=0.1)
    # np.save('/home/ryanzhu/V2I+V2V/Vehicular-Mininet-WiFi/ptcl/sample/6v_merged.npy', merged_pred)
    print('elapsed: ', time.time() - t_s)
    manager = multiprocessing.Manager()
    detected_spaces = manager.list()
    processes = []
    # start = time.time()
    for v_id in v_ids:
        p = multiprocessing.Process(target=get_detected_space, args=(merged_pred, detected_spaces,
                                                                     vehicle_pos[int(v_id)]))
        processes.append(p)
        p.start()
        # get_detected_space(merged_pred, detected_spaces, vehicle_pos[int(v_id)])
    for p in processes:
        p.join()
    # print('space detection takes:', time.time() - start)
    # merged_pred_grid = \
    #     ptcl.ptcl_utils.calculate_grid_label_ransac_new(1, merged_pred, center=vehicle_pos[int(v_id)])
    # detected_spaces.append(len(merged_pred_grid[merged_pred_grid != 0]))
    # print(detected_spaces)
    print('elapsed: ', time.time() - t_s)
    detected_spaces_list += detected_spaces
    return detected_spaces


if __name__ == '__main__':
    v_ids = [0, 1, 2, 3, 4, 5]
    frame_id = 0
    qb_dict = {0: 11, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11}
    detected_space_list = []
    time_start = time.time()
    # for frame_id in range(0, 10):
    calculate_merged_detection_spaces(v_ids, frame_id, qb_dict, detected_space_list)
    print('Passed:', time.time() - time_start)
