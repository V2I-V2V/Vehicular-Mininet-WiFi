import sys, os
# import ptcl.ground
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ptcl.pointcloud import dracoEncode, dracoDecode
import ptcl.ptcl_utils
import getpass
import numpy as np
import multiprocessing
import time

DATASET_DIR = '/home/'+getpass.getuser()+'/Carla/lidar/'

vehicle_id_to_dir = [86, 130, 174, 108, 119, 141, 152, 163, 185, 97]
vehicle_id_to_dir = [86, 97, 108, 119, 163, 141, 152, 163, 185, 97]

def get_detected_space(points, detected_spaces, detection_accuracy, grid_truth, center=(0, 0), local_pred = None):
    merged_pred_grid = \
            ptcl.ptcl_utils.calculate_grid_label_ransac_new(1, points, center=center)
    acc, _ = ptcl.ptcl_utils.calculate_precision(merged_pred_grid, grid_truth)
    if local_pred is None:
        detection_accuracy.append(acc)
    else:
        combined_grid = ptcl.ptcl_utils.combine_merged_results(local_pred, merged_pred_grid)
        # combined_grid = ptcl.ptcl_utils.combine_merged_results_on_remote(local_pred, merged_pred_grid)
        acc, _ = ptcl.ptcl_utils.calculate_precision(combined_grid, grid_truth)
        detection_accuracy.append(acc)
    detected_spaces.append(len(merged_pred_grid[merged_pred_grid != 0]))


def calculate_merged_detection_spaces(v_ids, frame_id, qb_dict, detected_spaces_list, acc_list):
    ptcls, vehicle_pos, grid_truth = [], {}, {}
    local_pred, local_points = {}, {}
    for v_id in v_ids:
        ptcl_name = DATASET_DIR + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)]) \
                    + '/' + str(1000+frame_id) 
        gt_file = DATASET_DIR + 'ground-truth-grid/' + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)])\
             + '-' + str(1000+frame_id) + '.txt'
        local_file = DATASET_DIR + 'local-prediction/' + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)])\
             + '-' + str(1000+frame_id) + '.txt'
        grid_truth[int(v_id)] = np.loadtxt(gt_file)
        pointcloud = ptcl.ptcl_utils.read_ptcl_data(ptcl_name + '.npy')
        local_points[int(v_id)] = pointcloud
        local_pred[int(v_id)] = np.loadtxt(local_file)
        trans = np.load(ptcl_name + '.trans.npy')
        encoded, _ = dracoEncode(pointcloud, 10, qb_dict[v_id])
        decoded = dracoDecode(encoded)
        pointcloud = np.concatenate([decoded, np.ones((decoded.shape[0], 1))], axis=1)
        pointcloud = np.dot(trans, pointcloud[:, :4].T).T
        ptcls.append(pointcloud)
        dummy = np.zeros((4, 1))
        dummy[3] = 1
        new_pos = np.dot(trans, dummy).T
        vehicle_pos[int(v_id)]= (new_pos[0,0], new_pos[0,1])
    merged = np.vstack(ptcls)
    merged_pred = ptcl.ptcl_utils.ransac_predict(merged, threshold=0.08)
    manager = multiprocessing.Manager()
    detected_spaces = manager.list()
    detection_accuracy = []
    processes = []
    # start = time.time()
    for v_id in v_ids:
        # p = multiprocessing.Process(target=get_detected_space, args=(merged_pred, detected_spaces, 
        #     vehicle_pos[int(v_id)]))
        # processes.append(p)
        # p.start()
        get_detected_space(merged_pred, detected_spaces, detection_accuracy, grid_truth[int(v_id)], \
            vehicle_pos[int(v_id)], local_pred[int(v_id)])
    for p in processes:
        p.join()
    # print('space detection takes:', time.time() - start)
        # merged_pred_grid = \
        #     ptcl.ptcl_utils.calculate_grid_label_ransac_new(1, merged_pred, center=vehicle_pos[int(v_id)])
        # detected_spaces.append(len(merged_pred_grid[merged_pred_grid != 0]))
    # print(detected_spaces)
    detected_spaces_list += detected_spaces
    acc_list += detection_accuracy
    return detected_spaces, detection_accuracy


if __name__ == '__main__':
    v_ids = [0, 1, 2, 3, 4, 5]
    frame_id = 0
    qb_dict = {0: 11, 1:11, 2:11, 3:11, 4:11, 5:11}
    detected_space_list = []
    time_start = time.time()
    calculate_merged_detection_spaces(v_ids, frame_id, qb_dict, detected_space_list)
    print('Passed:', time.time() - time_start)
