import numpy as np
import math
from ast import literal_eval
from analysis.v2i_bw import get_nodes_v2i_bw
from analysis.disconnection import get_disconect_duration_in_percentage

colors = ['r', 'b', 'maroon', 'darkblue', 'g', 'grey']

def get_server_assignments(filename):
    ts_to_assignment, ts_to_scores = {}, {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Assignment:"):
                # helper_to_helpee = {}
                helpee_to_helper = {}
                ts = float(line.split()[-1])
                assignment_str = line.split('(')[1].split(')')[0]
                assignment = eval(assignment_str)
                node_mapping_str = line.split('[')[1].split(']')[0]
                node_mapping_str = node_mapping_str.replace(' ', ', ')
                node_mapping = eval(node_mapping_str)
                for helpee_idx, helper_idx in enumerate(assignment):
                    helpee_to_helper[node_mapping[helpee_idx]] = node_mapping[helper_idx]
                ts_to_assignment[ts] = helpee_to_helper
            elif line.startswith('Scores: '):
                parse = line.split()
                ts = float(parse[-1])
                scores = (float(parse[1]), float(parse[2]), float(parse[3]))
                ts_to_scores[ts] = scores

    return ts_to_assignment, ts_to_scores


def get_sender_ts(filename):
    sender_ts = {}
    encode_choice = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parse = line.split()
            if line.startswith("[V2I"):
                ts = float(parse[-1])
                frame = int(parse[-2])
                sender_ts[frame] = ts
            elif line.startswith("[V2V"):
                ts = float(parse[-1])
                frame = int(parse[-5])
                sender_ts[frame] = ts
            elif line.startswith("frame id:"):
                num_chunks = int(parse[-1])
                frame = int(parse[2])
                encode_choice[frame] = num_chunks
            elif line.startswith("read and encode takes"):
                encode_t = math.ceil(float(parse[-1]))
            elif line.startswith("[relay throughput]"):
                last_t = float(parse[-1])
                
    return sender_ts, encode_choice, encode_t, last_t


def get_receiver_ts(filename):
    receiver_throughput = {}
    receiver_ts_dict = {}
    server_node_dict = {'time':[], 'helper_num': [], 'helpee_num': []}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("[Full frame recved]"):
                parse = line.split()
                sender_id = int(parse[4][:-1])
                frame = int(parse[6])
                thrpt = float(parse[8])
                ts = float(parse[-1])
                # receiver_ts_dict[sender_id].append(ts)
                if sender_id not in receiver_ts_dict.keys():
                    receiver_throughput[sender_id] = []
                    receiver_ts_dict[sender_id] = {}
                receiver_throughput[sender_id].append([ts, thrpt])
                receiver_ts_dict[sender_id][frame] = ts
            elif line.startswith("Helpers:"):
                parse = line.split()
                helper_cnt, helpee_cnt, ts = int(parse[1]), int(parse[3]), float(parse[-1])
                server_node_dict['time'].append(ts)
                server_node_dict['helper_num'].append(helper_cnt)
                server_node_dict['helpee_num'].append(helpee_cnt)
                
        f.close()
    return receiver_ts_dict, receiver_throughput, server_node_dict


def get_ssims(filename):
    ssims = np.loadtxt(filename)
    return ssims


def get_ssim(ssim_array, frame_id):
    idx = np.argwhere(ssim_array[:, 0] == frame_id)[0][0]
    return ssim_array[idx][1]


def get_helpees(helpee_conf):
    conf = np.loadtxt(helpee_conf)
    if len(conf) == 0:
        return np.array([])
    elif conf.ndim == 1:
        return np.array([conf[0]])
    else:
        return conf[0]


def get_num_frames_within_threshold(node_to_latency, threshold, ssim_t=None, perception_t=None):
    if ssim_t is None and perception_t is None:
        all_latency = node_to_latency['all']
        return len(all_latency[all_latency <= threshold])
    elif ssim_t is not None:
        cnt = 0
        for i in range(len(node_to_latency.keys())-3):
            node_stats = node_to_latency[i]
            within_latency_indices = np.array(sorted(node_stats.values()))[:, 0] < threshold
            within_ssim_indices = np.array(sorted(node_stats.values()))[:, 2] > ssim_t
            mask = within_latency_indices * within_ssim_indices
            cnt += len(np.where(mask==True)[0])
        return cnt


def get_percentage_frames_within_threshold(node_to_latency, threshold, ssim_t=None, perception_t=None):
    num_frames = get_num_frames_within_threshold(node_to_latency, threshold, ssim_t, perception_t)
    return num_frames/node_to_latency['sent_frames'] * 100.0


def get_stats_on_one_run(dir, num_nodes, helpee_conf, with_ssim=False):
    helpees = get_helpees(helpee_conf) # use a set, helpees represents all nodes that have been helpee
    sender_ts_dict, encode_choice_dict = {}, {}
    # key_to_value node_id_to_send_timestamps, node_id_to_encode_choices
    latency_dict, node_to_ssims, node_to_encode_choices = {}, {}, {}
    # node_id_to_latencies, node_id 
    sent_frames = 0
    for i in range(num_nodes):
        sender_ts_dict[i], node_to_encode_choices[i], encode_t, last_t = get_sender_ts(dir + '/logs/node%d.log'%i)
        sent_frames += int((last_t-min(sender_ts_dict[i].values()))*10)
        latency_dict[i] = {}
        if with_ssim:
            ssims = get_ssims(dir+'/node%d_ssim.log'%i)
            node_to_ssims[i] = ssims
    receiver_ts_dict, receiver_thrpt, server_helper_dict = get_receiver_ts(dir + '/logs/server.log')
    print("Total frames sent in exp", sent_frames)
    # calculate delay
    all_delay, helpee_delay, helper_delay = [], [], []
    full_frames = receiver_ts_dict[0].keys()
    for i in range(num_nodes):
        full_frames = full_frames & receiver_ts_dict[i].keys()
        for frame_idx, recv_ts in receiver_ts_dict[i].items():
            send_ts = sender_ts_dict[i][frame_idx]
            latency = recv_ts-sender_ts_dict[i][frame_idx]
            latency_dict[i][send_ts] = [latency, frame_idx] # add adptation choice
            if with_ssim:                
                latency_dict[i][send_ts] = [latency, frame_idx, get_ssim(node_to_ssims[i], frame_idx)]
            all_delay.append(latency)
            if i in helpees:
                helpee_delay.append(latency)
            else:
                helper_delay.append(latency)
    full_frame_delay = []
    for frame in full_frames:
        for i in range(num_nodes):
            full_frame_delay.append(receiver_ts_dict[i][frame]-sender_ts_dict[i][frame])
    
    latency_dict['all'] = np.array(all_delay)
    latency_dict['helpee'] = np.array(helpee_delay)
    latency_dict['helper'] = np.array(helper_delay)
    latency_dict['full_frames'] = np.array(full_frame_delay)
    latency_dict['sent_frames'] = sent_frames
    
    return latency_dict, node_to_encode_choices

def construct_ts_latency_array(delay_dict_ts, expected_frames=550):
    ts, delay = [], []
    last_frame_idx, idx_cnt = -1, 0
    sorted_ts = sorted(delay_dict_ts.keys())
    for send_ts in sorted_ts:
        frame_idx = delay_dict_ts[send_ts][1]
        if frame_idx > (last_frame_idx + 1):
            # skipped frames 
            print("skipped frmaes", frame_idx - last_frame_idx)
            # skipped_frames = frame_idx - last_frame_idx - 1
            last_ts = sorted_ts[idx_cnt-1]
            print("length", send_ts - last_ts)
            
            missed_tses = np.arange(last_ts, send_ts, 0.1)[1:]
            print(missed_tses)
            for missed_ts in missed_tses:
                ts.append(missed_ts)
                delay.append(-0.1)
        
        last_frame_idx = frame_idx
        idx_cnt += 1
        ts.append(send_ts)
        delay.append(delay_dict_ts[send_ts][0])
    
    # while len(ts) < expected_frames:
    #     ts.append(ts[-1]+0.1)
    #     delay.append(-1)
                       
    ts = np.array(ts) - np.min(ts)
    delay = np.array(delay)
    return ts, delay


def get_summary_of_settings(settings):
    for setting in settings:
        print("Get stats for setting", setting)
        num_nodes, sched, bw_file, loc, helpee_conf, run_time =\
            setting[0], setting[1], setting[2], setting[3], setting[4], setting[5]
        v2i_bw = get_nodes_v2i_bw(bw_file, run_time, num_nodes, helpee_conf)
        num_helpees, disconnect_percentage = \
            get_disconect_duration_in_percentage(helpee_conf, run_time, num_nodes)

        