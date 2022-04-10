import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_experiment import parse_config_from_file
import vehicle.octree
import numpy as np
import ptcl
import pickle


DATASET_DIR = '/home/mininet-wifi/dense_160_town05_far/lidar/'
vehicle_id_to_dir = [86, 97, 108, 119, 163, 141, 152, 130, 174, 185]
start_idx = 1000


def get_frame_ready_time(start_time, frame_id, fps=10):
    return start_time + 1/fps * frame_id


def get_carspeak_sender_stats(filename):
    sender_stats = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parse = line.split()
            if line.startswith('[V2I '):
                frame_id = int(parse[-3])
                timestamp = float(parse[-2])
                if frame_id not in sender_stats:
                    sender_stats[frame_id] = {'send_time': timestamp}
                else:
                    sender_stats[frame_id]['send_time'] = timestamp
            elif line.startswith('[start timestamp]'):
                start_timestamp = float(parse[-1])        
                sender_stats['start'] = start_timestamp
            elif line.startswith('[chunk '):
                frame_id, chunk_idx, v_id, timestamp = \
                        int(parse[-4]), int(parse[-3]), int(parse[-2]), float(parse[-1])
                if frame_id not in sender_stats:
                    sender_stats[frame_id] = {'recv_chunks': [(chunk_idx, v_id, timestamp)]}
                else:
                    if 'recv_chunks' not in sender_stats[frame_id]:
                        sender_stats[frame_id]['recv_chunks'] = [(chunk_idx, v_id, timestamp)]
                    else:
                        sender_stats[frame_id]['recv_chunks'].append((chunk_idx, v_id, timestamp))
            elif line.startswith('[All possible frame recved]'):
                frame_id, timestamp = int(parse[-2]), float(parse[-1])
                if frame_id not in sender_stats:
                    sender_stats[frame_id] = {'finish-latency': timestamp - (start_timestamp + 0.1 * frame_id)}
                sender_stats[frame_id]['finish-latency'] = timestamp - (start_timestamp + 0.1 * frame_id)

    return sender_stats


def load_ptcls(frame_id, ego_v_id, num_vehicles=2):
    ptcls, vehicle_pos = [], {}
    for v_id in range(len(vehicle_id_to_dir[:num_vehicles])):
        ptcl_name = DATASET_DIR + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)]) \
                + '/' + str(start_idx+frame_id) 
        gt_file = DATASET_DIR + 'ground-truth-grid/' + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)])\
                    + '-' + str(start_idx+frame_id) + '.txt'
        local_file = DATASET_DIR + 'local-prediction/' + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)])\
                    + '-' + str(start_idx+frame_id) + '.txt'
        pointcloud = ptcl.ptcl_utils.read_ptcl_data(ptcl_name + '.npy')      
        trans = np.load(ptcl_name + '.trans.npy')
        if ego_v_id == v_id:
            ptcls.append(np.dot(trans, pointcloud[:, :4].T).T)
            grid_truth = np.loadtxt(gt_file)
            local_pred= np.loadtxt(local_file)
        else:
            oct = vehicle.octree.OcTree(pointcloud, depth=7)
            oct.encode()
            oct.decode()
            pointcloud = np.concatenate([oct.decoded, np.ones((oct.decoded.shape[0], 1))], axis=1)
            ptcls.append(np.dot(trans, pointcloud[:, :4].T).T)
        dummy = np.zeros((4, 1))
        dummy[3] = 1
        new_pos = np.dot(trans, dummy).T
        vehicle_pos[int(v_id)]= (new_pos[0,0], new_pos[0,1])
    return ptcls, vehicle_pos, grid_truth, local_pred


def calculate_detection_acc_carspeak(ptcls, ego_v_id, local_pred, grid_truth, \
    center=(0,0), recv_chunks=None, mode='all'):
    if mode == 'all':
        merged = np.vstack(ptcls)
        merged_pred = ptcl.ptcl_utils.ransac_predict(merged, threshold=0.08)
        merged_pred_grid = \
            ptcl.ptcl_utils.calculate_grid_label_ransac_new(1, merged_pred, center=center)
        acc, _ = ptcl.ptcl_utils.calculate_precision(merged_pred_grid, grid_truth)
    elif mode == 'local':
        acc = ptcl.ptcl_utils.calculate_precision(local_pred, grid_truth)[0]
    elif mode == 'partial':
        final_ptcl = [ptcls[ego_v_id]]
        for chunk in recv_chunks:
            chunk_id, v_id, timestamp = chunk
            if (chunk_id+1) * 500 >= ptcls[v_id].shape[0]:
                final_ptcl.append(ptcls[v_id][chunk_id*500:])
            else:
                final_ptcl.append(ptcls[v_id][chunk_id*500:(chunk_id+1)*500])
                # print("chunk size", ptcls[v_id][chunk_id*500:(chunk_id+1)*500].shape)
        merged = np.vstack(final_ptcl)
        merged_pred = ptcl.ptcl_utils.ransac_predict(merged, threshold=0.08)
        merged_pred_grid = \
            ptcl.ptcl_utils.calculate_grid_label_ransac_new(1, merged_pred, center=center)
            
        acc, _ = ptcl.ptcl_utils.calculate_precision(merged_pred_grid, grid_truth)
    return acc


def get_metrics(sender_stats, sender_id, num_nodes, latency_threshold=0.5):
    latencies = {}
    ptcls_acc = {}
    max_id = 0
    for f_id, rst in sender_stats.items():
        if type(f_id) is int:
            if f_id > max_id:
                max_id = f_id
            ptcls, vehicle_pos, grid_truth, local_pred = load_ptcls(f_id % 300, sender_id, num_nodes)
            # print(sender_id, len(ptcls), vehicle_pos[sender_id])
            if 'finish-latency' in rst and rst['finish-latency'] <= latency_threshold:
                latencies[f_id] = rst['finish-latency']
                acc = calculate_detection_acc_carspeak(ptcls, sender_id, local_pred, grid_truth,
                                                        center=vehicle_pos[sender_id], mode='all')  
                ptcls_acc[f_id] = acc          
            else:
                max_delay = 0
                if 'recv_chunks' in rst:
                    valid_chunks = []
                    for chunk in rst['recv_chunks']:
                        chunk_id, v_id, timestamp = chunk
                        if timestamp - get_frame_ready_time(sender_stats['start'], f_id) > latency_threshold:
                            max_delay = latency_threshold
                        elif timestamp - get_frame_ready_time(sender_stats['start'], f_id) > max_delay:
                            max_delay = timestamp - get_frame_ready_time(sender_stats['start'], f_id)
                        if timestamp - get_frame_ready_time(sender_stats['start'], f_id) <= latency_threshold:
                            # print('valid chunks', chunk_id, v_id)
                            valid_chunks.append(chunk)
                    latencies[f_id] = max_delay
                    if len(valid_chunks) > 0:
                        acc = calculate_detection_acc_carspeak(ptcls, sender_id, local_pred, grid_truth,
                                                            center=vehicle_pos[sender_id], recv_chunks=valid_chunks,
                                                            mode='partial')   
                        ptcls_acc[f_id] = acc  
                    else:
                        acc = calculate_detection_acc_carspeak(ptcls, sender_id, local_pred, grid_truth,
                                                        center=vehicle_pos[sender_id], mode='local')
                    # print(acc)
                else:
                    latencies[f_id] = latency_threshold
                    acc = calculate_detection_acc_carspeak(ptcls, sender_id, local_pred, grid_truth,
                                                        center=vehicle_pos[sender_id], mode='local')
                    ptcls_acc[f_id] = acc  
    
    for frame in range(max_id):
        if frame not in sender_stats:
            ptcls, vehicle_pos, grid_truth, local_pred = load_ptcls(frame % 300, sender_id, num_nodes)
            latencies[frame] = latency_threshold
            acc = calculate_detection_acc_carspeak(ptcls, sender_id, local_pred, grid_truth,
                                                    center=vehicle_pos[sender_id], mode='local')
            ptcls_acc[frame] = acc  
    
    return latencies, ptcls_acc


def get_single_run_stats(dir, num_nodes):
    print(dir)
    sender_stats_summary = {}
    result_summary = {}
    for i in range(num_nodes):
        if os.path.exists(dir + '/logs/node%d.log'%i):
            sender_stats_summary[i] = get_carspeak_sender_stats(dir + '/logs/node%d.log'%i)
            latency, ptcls_acc = get_metrics(sender_stats_summary[i], i, num_nodes, latency_threshold=0.3)
            print(ptcls_acc.values())
            if 'latency' not in result_summary:
                result_summary['latency'] = [latency]
            else:
                 result_summary['latency'].append(latency)
            if 'acc' not in result_summary:
                result_summary['acc'] = [ptcls_acc]
            else:
                 result_summary['acc'].append(ptcls_acc)
    with open(dir+'/summary_0.3.pickle', 'wb') as s:
        pickle.dump(result_summary, s)


if __name__ == '__main__':
    exp_rst_folder = '/home/mininet-wifi/v2x_exp_comprehensive/carspeak_new/'
    data_folders = os.listdir(exp_rst_folder)
    for fold in data_folders:
        if 'data' in fold:
            print(fold)
            config = parse_config_from_file(exp_rst_folder + fold + '/config.txt')
        # config = parse_config_from_file('/home/mininet-wifi/v2x_exp_comprehensive/carspeak/data-03301835/config.txt')
            if 'carla-town05-160' in config['ptcl_config']:
                DATASET_DIR = '/home/mininet-wifi/dense_160_town05_far/lidar/'
                vehicle_id_to_dir = [202, 249, 255, 259, 289, 298, 310, 327, 347, 356]
                start_idx = 800
            elif 'carla-town05-120' in config['ptcl_config']:
                DATASET_DIR = '/home/mininet-wifi/dense_120_town05/lidar/'
                vehicle_id_to_dir = vehicle_id_to_dir = [209, 221, 233, 269, 270, 295, 152, 163, 185, 97]
                start_idx = 1000
            elif 'carla-data-config' in config['ptcl_config']:
                print('carla data config!!')
                DATASET_DIR = '/home/mininet-wifi/Carla/lidar/'
                vehicle_id_to_dir = [86, 97, 108, 119, 163, 141, 152, 130, 174, 185]
                start_idx = 1000
            elif 'carla-town03-120-1' in config['ptcl_config']:
                DATASET_DIR = '/home/mininet-wifi/dense_120_town03_1/lidar/'
                vehicle_id_to_dir = [971, 1026, 1027, 1033, 1070, 1083, 310, 327, 347, 356]
                start_idx = 752
            get_single_run_stats(exp_rst_folder + fold, int(config["num_of_nodes"]))
        
