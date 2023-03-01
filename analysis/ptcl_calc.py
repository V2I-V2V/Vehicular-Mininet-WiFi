import sys, os
# import ptcl.ground
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ptcl.pointcloud import dracoEncode, dracoDecode
import ptcl.ptcl_utils
import getpass
import numpy as np
import multiprocessing
import time
import math

# DATASET_DIR = '/home/mininet-wifi/Carla/lidar/'
# DATASET_DIR = '/home/mininet-wifi/dense_120_town05/lidar/'
DATASET_DIR = '/home/mininet-wifi/dense_160_town05_far/lidar/'
DATASET_DIR = '/home/mininet-wifi/6-cav-22-non/lidar/'
# DATASET_DIR = '/home/mininet-wifi/dense_120_town03_1/lidar/'
vehicle_deadline = 0.5
start_f_id = 64


vehicle_id_to_dir = [86, 130, 174, 108, 119, 141, 152, 163, 185, 97]
vehicle_id_to_dir = [86, 97, 108, 119, 163, 141, 152, 130, 174, 185]

vehicle_id_to_dir = [86, 97, 108, 119, 163, 141, 152, 130, 174, 185]
vehicle_id_to_dir = [209, 221, 233, 269, 270, 295, 152, 163, 185, 97]
vehicle_id_to_dir = [202, 249, 255, 259, 289, 298, 310, 327, 347, 356]
# vehicle_id_to_dir = [971, 1026, 1027, 1033, 1070, 1083, 310, 327, 347, 356]
vehicle_id_to_dir = [256, 257, 258, 259, 260, 261]



def get_detected_space(points, detected_spaces, detection_accuracy, grid_truth, center=(0, 0), local_pred=None):
    global remote_acc
    merged_pred_grid = \
            ptcl.ptcl_utils.calculate_grid_label_ransac_new(1, points, center=center)
    if local_pred is None:
        acc, _ = ptcl.ptcl_utils.calculate_precision(merged_pred_grid, grid_truth)
        detection_accuracy.append(acc)
    else:
        combined_grid = ptcl.ptcl_utils.combine_merged_results(local_pred, merged_pred_grid)
        # combined_grid = ptcl.ptcl_utils.combine_merged_results_on_remote(local_pred, merged_pred_grid)
        acc, _ = ptcl.ptcl_utils.calculate_precision(combined_grid, grid_truth)
        # print(ptcl.ptcl_utils.calculate_precision(local_pred, grid_truth)[0])
        # acc = ptcl.ptcl_utils.calculate_oracle_accuracy(local_pred, merged_pred_grid, grid_truth)
        detection_accuracy.append(acc)
        local_acc, _ = ptcl.ptcl_utils.calculate_precision(local_pred, grid_truth)
        # print('local', local_acc, center)
        remote_acc, _ = ptcl.ptcl_utils.calculate_precision(merged_pred_grid, grid_truth)
        remote_accs.append(remote_acc)
        # print('remote', remote_acc, center)
        oracle = ptcl.ptcl_utils.calculate_oracle_accuracy(local_pred, merged_pred_grid, grid_truth)
        print(local_acc, remote_acc, acc, oracle)
        oracle_accs.append(oracle)
    detected_spaces.append(len(merged_pred_grid[merged_pred_grid != 0]))


def calculate_merged_detection_spaces(v_ids, frame_id, qb_dict, detected_spaces_list, acc_list, scheme, num_nodes, nodes_latency, dataset_dir='/home/mininet-wifi/dense_120_town05/lidar/'):
    # print(v_ids, frame_id, qb_dict, nodes_latency, scheme)
    # print(qb_dict)

    accuracy = []
    for g in range(math.ceil(num_nodes/10)):
        ptcls, vehicle_pos, grid_truth = [], {}, {}
        local_pred, local_points = {}, {}
        if g == math.ceil(num_nodes/10) - 1:
            if num_nodes % 10 != 0:
                selected_vids = np.arange(g*10,g*10+ num_nodes % 10)
                print(selected_vids)
            else:
                selected_vids = np.arange(g*10, (g+1)*10)
        else:
            selected_vids = np.arange(g*10, (g+1)*10)
        # for v_id in selected_vids:
            # if str(v_id) in v_ids:
        for v_id in v_ids:
            if int(v_id) in selected_vids:
                ptcl_name = DATASET_DIR + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)]) \
                            + '/' + str(start_f_id+frame_id) 
                gt_file = DATASET_DIR + 'ground-truth-grid/' + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)])\
                    + '-' + str(start_f_id+frame_id) + '.txt'
                local_file = DATASET_DIR + 'local-prediction/' + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)])\
                    + '-' + str(start_f_id+frame_id) + '-local.txt'
                grid_truth[int(v_id)] = np.loadtxt(gt_file)
                pointcloud = ptcl.ptcl_utils.read_ptcl_data(ptcl_name + '.npy')
                local_points[int(v_id)] = pointcloud
                local_pred[int(v_id)] = np.loadtxt(local_file)
                trans = np.load(ptcl_name + '.trans.npy')
                encoded, _ = dracoEncode(pointcloud, 10, qb_dict[str(v_id)])
                decoded = dracoDecode(encoded)
                pointcloud = np.concatenate([decoded, np.ones((decoded.shape[0], 1))], axis=1)
                pointcloud = np.dot(trans, pointcloud[:, :4].T).T
                ptcls.append(pointcloud)
                dummy = np.zeros((4, 1))
                dummy[3] = 1
                new_pos = np.dot(trans, dummy).T
                vehicle_pos[int(v_id)]= (new_pos[0,0], new_pos[0,1])
        for v_id in selected_vids:    
            # load every gt and local
            gt_file = DATASET_DIR + 'ground-truth-grid/' + str(vehicle_id_to_dir[v_id%len(vehicle_id_to_dir)])\
                + '-' + str(start_f_id+frame_id) + '.txt'
            local_file = DATASET_DIR + 'local-prediction/' + str(vehicle_id_to_dir[v_id%len(vehicle_id_to_dir)])\
                + '-' + str(start_f_id+frame_id) + '-local.txt'
            grid_truth[v_id] = np.loadtxt(gt_file)
            local_pred[v_id] = np.loadtxt(local_file)
        if len(ptcls) > 0:
            merged = np.vstack(ptcls)
            merged_pred = ptcl.ptcl_utils.ransac_predict(merged, threshold=0.08)
        detected_spaces = []
        detection_accuracy = []
        for v_id in selected_vids:        
            if str(v_id) in v_ids and nodes_latency[v_id] < vehicle_deadline:
                if '-combined' in scheme:
                    get_detected_space(merged_pred, detected_spaces, detection_accuracy, grid_truth[int(v_id)], \
                        vehicle_pos[int(v_id)], local_pred=local_pred[int(v_id)])
                else:
                    get_detected_space(merged_pred, detected_spaces, detection_accuracy, grid_truth[int(v_id)], \
                        vehicle_pos[int(v_id)], local_pred=None)
            else:
                # use local detection instead
                detection_accuracy.append(ptcl.ptcl_utils.calculate_precision(local_pred[int(v_id)], grid_truth[int(v_id)])[0])

        if len(detected_spaces) > 0:
            detected_spaces_list += detected_spaces
        if len(detection_accuracy) == 0:
            print("error detection acc empty")
        accuracy.extend(detection_accuracy)
    
    acc_list.extend(accuracy)
    # for v_id in v_ids:
    #     ptcl_name = DATASET_DIR + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)]) \
    #                 + '/' + str(start_f_id+frame_id) 
    #     gt_file = DATASET_DIR + 'ground-truth-grid/' + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)])\
    #          + '-' + str(start_f_id+frame_id) + '.txt'
    #     local_file = DATASET_DIR + 'local-prediction/' + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)])\
    #          + '-' + str(start_f_id+frame_id) + '.txt'
    #     grid_truth[int(v_id)] = np.loadtxt(gt_file)
    #     pointcloud = ptcl.ptcl_utils.read_ptcl_data(ptcl_name + '.npy')
    #     local_points[int(v_id)] = pointcloud
    #     local_pred[int(v_id)] = np.loadtxt(local_file)
    #     trans = np.load(ptcl_name + '.trans.npy')
    #     encoded, _ = dracoEncode(pointcloud, 10, qb_dict[v_id])
    #     decoded = dracoDecode(encoded)
    #     pointcloud = np.concatenate([decoded, np.ones((decoded.shape[0], 1))], axis=1)
    #     pointcloud = np.dot(trans, pointcloud[:, :4].T).T
    #     ptcls.append(pointcloud)
    #     dummy = np.zeros((4, 1))
    #     dummy[3] = 1
    #     new_pos = np.dot(trans, dummy).T
    #     vehicle_pos[int(v_id)]= (new_pos[0,0], new_pos[0,1])
    # for v_id in range(num_nodes):    
    #     # load every gt and local
    #     gt_file = DATASET_DIR + 'ground-truth-grid/' + str(vehicle_id_to_dir[v_id%len(vehicle_id_to_dir)])\
    #          + '-' + str(start_f_id+frame_id) + '.txt'
    #     local_file = DATASET_DIR + 'local-prediction/' + str(vehicle_id_to_dir[v_id%len(vehicle_id_to_dir)])\
    #          + '-' + str(start_f_id+frame_id) + '.txt'
    #     grid_truth[v_id] = np.loadtxt(gt_file)
    #     local_pred[v_id] = np.loadtxt(local_file)
    # if len(ptcls) > 0:
    #     merged = np.vstack(ptcls)
    #     merged_pred = ptcl.ptcl_utils.ransac_predict(merged, threshold=0.08)
    # detected_spaces = []
    # detection_accuracy = []
    # for v_id in range(num_nodes):        
    #     if str(v_id) in v_ids and nodes_latency[v_id] < vehicle_deadline:
    #         if 'combined' in scheme:
    #             get_detected_space(merged_pred, detected_spaces, detection_accuracy, grid_truth[int(v_id)], \
    #                 vehicle_pos[int(v_id)], local_pred=local_pred[int(v_id)])
    #         else:
    #             get_detected_space(merged_pred, detected_spaces, detection_accuracy, grid_truth[int(v_id)], \
    #                 vehicle_pos[int(v_id)], local_pred=None)
    #     else:
    #         # use local detection instead
    #         detection_accuracy.append(ptcl.ptcl_utils.calculate_precision(local_pred[int(v_id)], grid_truth[int(v_id)])[0])
            
    # # print('space detection takes:', time.time() - start)
    #     # merged_pred_grid = \
    #     #     ptcl.ptcl_utils.calculate_grid_label_ransac_new(1, merged_pred, center=vehicle_pos[int(v_id)])
    #     # detected_spaces.append(len(merged_pred_grid[merged_pred_grid != 0]))
    # if len(detected_spaces) > 0:
    #     detected_spaces_list += detected_spaces
    # if len(detection_accuracy) == 0:
    #     print("error detection acc empty")
    # # print(detection_accuracy)
    # acc_list.extend(detection_accuracy)
    return detected_spaces, detection_accuracy


def calculate_local_detection_spaces(v_id, frame_id):
    gt_file = DATASET_DIR + 'ground-truth-grid/' + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)])\
             + '-' + str(start_f_id+frame_id) + '.txt'
    local_file = DATASET_DIR + 'local-prediction/' + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)])\
             + '-' + str(start_f_id+frame_id) + '-local.txt'
    grid_truth = np.loadtxt(gt_file)
    local_pred = np.loadtxt(local_file)
    return ptcl.ptcl_utils.calculate_precision(local_pred, grid_truth)[0]

if __name__ == '__main__':
    v_ids = ['0', '1', '2', '3']
    frame_id = 0
    qb_dict = {'0': 11, '1': 11, '2': 11, '3': 11}
    nodes_latency = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}
    detected_space_list, acc_list = [], []
    time_start = time.time()
    calculate_merged_detection_spaces(v_ids, frame_id, qb_dict, detected_space_list, acc_list, 'combined', len(v_ids),
                                      nodes_latency)
    print(acc_list)
    print('Passed:', time.time() - time_start)
    print(acc_list)
    print(np.mean(acc_list), np.mean(remote_accs), np.mean(oracle_accs))
