import collections
from typing import overload
import numpy as np
import math
from ast import literal_eval
from analysis.v2i_bw import get_nodes_v2i_bw
from analysis.disconnection import get_disconect_duration_in_percentage
from analysis.trajectory import get_node_dists
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import os

colors = ['r', 'b', 'maroon', 'darkblue', 'g', 'grey']

sched_to_color = {'minDist': 'r', 'random': 'b', 'distributed': 'maroon', 'combined': 'g',\
    'combined-adapt': 'grey', 'bwAware': 'darkblue', 'combined-op_min-min': 'blueviolet',
    'combined-loc': 'brown', 'combined-op_sum-min': 'darkorange',
    'combined-op_sum-harmonic': 'cyan', 'emp': 'orange'}
sched_to_line_style = {'minDist': '', 'random': ' ', 'distributed': '--', 'combined': ':',\
    'combined-adapt': '-'}

linestyles = OrderedDict(
    [('combined-adapt',               (0, ())),
     ('minDist',      (0, (1, 10))),
     ('combined-op_min-min',              (0, (1, 5))),
     ('combined',      (0, (1, 1))),

     ('combined-op_sum-min',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('random',      (0, (5, 1))),
     ('emp',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('distributed',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('combined-op_sum-harmonic', (0, (3, 10, 1, 10, 1, 10))),
     ('combined-loc',         (0, (3, 5, 1, 5, 1, 5))),
     ('bwAware', (0, (3, 1, 1, 1, 1, 1)))])


detected_spaces_label = [[[] for _ in range(80)] for _ in range(6)]
detected_space_label = collections.defaultdict(list)

# for i in range(1, 7):
#     for j in range(80):
#         grid_label = np.loadtxt('/home/mininet-wifi/all_grid_labels/%06d_%d.txt'%(j, i))
#         detected_space = len(grid_label[grid_label != 0])
#         detected_spaces_label[i-1][j] = detected_space

# labels = os.listdir('/home/mininet-wifi/grid_labels_all_comb/')
# for label in sorted(labels):
#     path = '/home/mininet-wifi/grid_labels_all_comb/' + label
#     grids = np.loadtxt(path)
#     detected_space = len(grids[grids != 0])
#     comb = label[7:-4]
#     detected_space_label[comb].append(detected_space)
import json
# with open("/home/mininet-wifi/all-grid-label.json", 'w') as outfile:
#     json.dump(detected_space_label, outfile)
f = open("/home/mininet-wifi/all-grid-label.json", 'r')
detected_space_label = json.load(f)


def get_server_assignments(filename):
    ts_to_assignment, ts_to_scores = {}, {}
    score_combined = {'comb': [], 'min': [], 'ass_combined': [], 'ass_min': []}
    score_min = {'min': [], 'comb': [], 'ass_combined': [], 'ass_min': []}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Run in'):
                assignment_mode = line.split()[2]
                if assignment_mode == 'fix':
                    return assignment_mode, ts_to_assignment, ts_to_scores, score_combined, score_min
            if line.startswith("Assignment:"):
                # helper_to_helpee = {}
                helpee_to_helper = {}
                ts = float(line.split()[-1])
                assignment_str = line.split('(')[1].split(')')[0]
                if len(assignment_str) != 0:
                    assignment = eval(assignment_str)
                else:
                    assignment = ()
                node_mapping_str = line.split('[')[1].split(']')[0]
                node_mapping_str = node_mapping_str.replace(' ', ', ')
                # node_mapping = eval(node_mapping_str)
                if len(node_mapping_str) != 0:
                    node_mapping = eval(node_mapping_str)
                try:
                    for helpee_idx, helper_idx in enumerate(assignment):
                        # print(node_mapping, helpee_idx)
                        helpee_to_helper[node_mapping[helpee_idx]] = node_mapping[helper_idx]
                    ts_to_assignment[ts] = helpee_to_helper
                except:
                    pass
            elif line.startswith('Scores '):
                parse = line.split()
                ts = float(parse[-1])
                # first 6 scores are harmonic values, last 6 scores are min values
                scores_harmonic_min = (float(parse[2][1:-1]), float(parse[3][:-1]), float(parse[4][1:-1]), \
                    float(parse[5][:-1]), float(parse[6][1:-1]), float(parse[7][:-1]), \
                    float(parse[8][1:-1]), float(parse[9][:-1]), float(parse[10][1:-1]), \
                    float(parse[11][:-1]), float(parse[12][1:-1]), float(parse[13][:-1]))
                
                
                ts_to_scores[ts] = scores_harmonic_min
            elif line.startswith('Best choice (min_sum/combined) scores:'):
                parse = line.split()
                ass1, ass2 = parse[-6], parse[-5]
                score_best_combined_min1, score_combine_on_min = float(parse[-4]), float(parse[-3])
                score_best_combined1, score_combined_min_on_combined = float(parse[-1]), float(parse[-2])
                score_min['min'].append(score_best_combined_min1)
                score_min['comb'].append(score_combine_on_min)
                score_combined['comb'].append(score_best_combined1)
                score_combined['min'].append(score_combined_min_on_combined)
                score_combined['ass_combined'].append(ass2)
                score_min['ass_min'].append(ass1)
                

    return assignment_mode, ts_to_assignment, ts_to_scores, score_combined, score_min


def get_computation_overhead(filename):
    computation_latency = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parse = line.split() 
            if line.startswith('[start timestamp]'):
                t1 = float(parse[-1])
            elif line.startswith('[Loc msg size]') or line.startswith('[Loc msg size]'):
                t1 = float(parse[-1])
            elif line.startswith('[Helpee get helper assignment]'):
                t5 = float(parse[-1])
                computation_latency.append(t5-t1)
    return computation_latency


def get_sender_ts(filename):
    sender_ts = {}
    encode_choice = {}
    summary_dict = {'dl-latency': [], 'e2e-latency': []} # sumamry of related metrics
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parse = line.split()
            if line.startswith('[start timestamp]'):
                start_ts = float(parse[-1])
            elif line.startswith('fps '):
                fps = int(parse[-1])
            elif line.startswith("[V2I"):
                frame = int(parse[-2])
                sender_ts[frame] = frame * 1.0/fps + start_ts
                last_t = float(parse[-1])
            elif line.startswith("[V2V send"):
                frame = int(parse[-5])
                sender_ts[frame] = frame * 1.0/fps + start_ts
                last_t = float(parse[-1])
            elif line.startswith("frame id:"):
                num_chunks = int(parse[-1])
                frame = int(parse[2])
                encode_choice[frame] = num_chunks
                last_t = float(parse[-1])
            elif line.startswith("read and encode takes"):
                encode_t = math.ceil(float(parse[-1]))
            elif line.startswith("[relay throughput]"):
                last_t = float(parse[-1])
            elif line.startswith("[Recv rst from server]") \
                or line.startswith("[V2V Recv rst from helper]"):
                summary_dict['dl-latency'].append(float(parse[-2]))
                timestamp = float(parse[-1])
                frame = int(parse[-5][:-1])
                e2e_latency = timestamp - (frame * 1.0/fps + start_ts)
                # if frame in sender_ts:
                #     e2e_latency = timestamp - sender_ts[frame]
                summary_dict['e2e-latency'].append(e2e_latency)
                
    return sender_ts, encode_choice, encode_t, last_t, summary_dict


def get_receiver_ts(filename):
    receiver_throughput = {}
    receiver_ts_dict = {}
    server_node_dict = {'time':[], 'helper_num': [], 'helpee_num': []}
    sched_latencies = []
    frame_id_to_senders = defaultdict(str)
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
                if str(sender_id) not in frame_id_to_senders[frame]:
                    frame_id_to_senders[frame] += str(sender_id)
                # if frame_id_to_senders[frame] < 6:
                #     frame_id_to_senders[frame] += 1
            elif line.startswith("Helpers:"):
                parse = line.split()
                helper_cnt, helpee_cnt, ts = int(parse[1]), int(parse[3]), float(parse[-1])
                server_node_dict['time'].append(ts)
                server_node_dict['helper_num'].append(helper_cnt)
                server_node_dict['helpee_num'].append(helpee_cnt)
            elif line.startswith("One round sched takes"):
                sched_latency = float(line.split()[-1])
                sched_latencies.append(sched_latency)
                
        f.close()
    return receiver_ts_dict, receiver_throughput, server_node_dict, sched_latencies, frame_id_to_senders


def construct_comb(vnum_list, truth_list):
    s = ""
    for i in vnum_list:
        s += truth_list[int(i)]
    return s

def calculate_detected_areas(frame_id_to_senders):
    detected_spaces = []
    for frame_id, v_num in frame_id_to_senders.items():
        if frame_id < 550:
            wrapped_frame_id = frame_id % 80
            print(v_num)
            v_num_comb = sorted(v_num)
            key = construct_comb(v_num_comb, ['0', '2', '4', '5'])
            # grid_label = np.loadtxt('/home/mininet-wifi/all_grid_labels/%06d_%d.txt'%(wrapped_frame_id, v_num))
            # detected_space = len(grid_label[grid_label != 0])
            # detected_spaces.append(detected_space)
            print(type(wrapped_frame_id))
            detected_spaces.append(detected_space_label[key][wrapped_frame_id])
    return detected_spaces


def get_distributed_helper_assignments(data_dir, num_nodes):
    assignment_mode, ts_to_assignment = 'distributed', {}
    helpee_to_helper = {}
    for i in range(num_nodes):
        filename = data_dir + '/logs/node%d.log'%i
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("[Helpee decide to use helper assignment]"):
                    parse = line.split()
                    helper, ts = int(parse[-2]), float(parse[-1])
                    # helpee_to_helper = {i: helper}
                    helpee_to_helper[i] = helper
                    break

    ts_to_assignment[ts] = helpee_to_helper
    ts_to_assignment[ts+56] = helpee_to_helper

    return assignment_mode, ts_to_assignment


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


def get_num_frames_above_detected_space_threshold(detected_spaces, threshold):
    return len(detected_spaces[detected_spaces >= threshold])


def get_num_frames_within_threshold(node_to_latency, threshold, ssim_t=None, perception_t=None, key='all'):
    if ssim_t is None and perception_t is None:
        all_latency = node_to_latency[key]
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


def get_percentage_frames_within_threshold(node_to_latency, threshold, ssim_t=None, perception_t=None, key='all',\
    num_nodes=6):
    num_frames = get_num_frames_within_threshold(node_to_latency, threshold, ssim_t, perception_t, key=key)
    if key == 'max_full_frames':
        return num_frames/node_to_latency['sent_frames'] * 100.0 * num_nodes
    else:
        return num_frames/node_to_latency['sent_frames'] * 100.0


def get_stats_on_one_run(dir, num_nodes, helpee_conf, with_ssim=False):
    helpees = get_helpees(helpee_conf) # use a set, helpees represents all nodes that have been helpee
    sender_ts_dict, encode_choice_dict = {}, {}
    # key_to_value node_id_to_send_timestamps, node_id_to_encode_choices
    latency_dict, node_to_ssims, node_to_encode_choices = {}, {}, {}
    # node_id_to_latencies, node_id 
    overhead, dl_latencies = [], {}
    sent_frames = 0
    for i in range(num_nodes):
        sender_ts_dict[i], node_to_encode_choices[i], encode_t, last_t, summary_dict = get_sender_ts(dir + '/logs/node%d.log'%i)
        # sent_frames += int((last_t-min(sender_ts_dict[i].values()))*10)
        computational_overhead = get_computation_overhead(dir + '/logs/node%d.log'%i)
        # dl_latencies.extend(summary_dict['dl-latency'])
        dl_latencies[i] = summary_dict['dl-latency']
        if len(computational_overhead) > 0:
            print("overhead", np.mean(computational_overhead))
            overhead.extend(computational_overhead)

        sent_frames += 556
        latency_dict[i] = {}
        if with_ssim:
            ssims = get_ssims(dir+'/node%d_ssim.log'%i)
            node_to_ssims[i] = ssims
    receiver_ts_dict, receiver_thrpt, server_helper_dict, sched_latencies, frame_id_to_senders = get_receiver_ts(dir + '/logs/server.log')
    detected_areas = calculate_detected_areas(frame_id_to_senders)
    # detected_areas = None
    print("Total frames sent in exp", sent_frames)
    # calculate delay
    all_delay, helpee_delay, helper_delay = [], [], []
    full_frames = receiver_ts_dict[0].keys()
    for i in range(num_nodes):
        full_frames = full_frames & receiver_ts_dict[i].keys()
        for frame_idx, recv_ts in receiver_ts_dict[i].items():
            send_ts = sender_ts_dict[i][frame_idx]
            latency = recv_ts-sender_ts_dict[i][frame_idx]
            # if latency < 0:
            #     print("negative latency! %f, %d"%(latency, frame_idx))
            #     print(dir)
            #     print(i)
            #     exit(1)
            latency_dict[i][send_ts] = [latency, frame_idx] # add adptation choice
            if with_ssim:                
                latency_dict[i][send_ts] = [latency, frame_idx, get_ssim(node_to_ssims[i], frame_idx)]
            all_delay.append(latency)
            if i in helpees:
                helpee_delay.append(latency)
            else:
                helper_delay.append(latency)
    full_frame_delay, full_frame_max_delay = [], []
    for frame in full_frames:
        for i in range(num_nodes):
            full_frame_delay.append(receiver_ts_dict[i][frame]-sender_ts_dict[i][frame])
        full_frame_max_delay.append(max(full_frame_delay[-num_nodes:]))
    for frame_id in frame_id_to_senders:
        delay = []
        for i in range(num_nodes):
            if frame_id in receiver_ts_dict[i]:
                delay.append(receiver_ts_dict[i][frame_id]-sender_ts_dict[i][frame_id])
        # full_frame_max_delay.append(max(delay))
    latency_dict['all'] = np.array(all_delay)
    latency_dict['helpee'] = np.array(helpee_delay)
    latency_dict['helper'] = np.array(helper_delay)
    latency_dict['full_frames'] = np.array(full_frame_delay)
    latency_dict['sent_frames'] = sent_frames
    latency_dict['max_full_frames'] = np.array(full_frame_max_delay)
    latency_dict['overhead'] = np.array(overhead)
    latency_dict['sched_latency'] = np.array(sched_latencies)
    latency_dict['detected_areas'] = detected_areas
    latency_dict['dl-latency'] = dl_latencies
    
    return latency_dict, node_to_encode_choices

def construct_ts_latency_array(delay_dict_ts, expected_frames=550):
    ts, delay = [], []
    last_frame_idx, idx_cnt = -1, 0
    sorted_ts = sorted(delay_dict_ts.keys())
    skipped_frames = 0
    for send_ts in sorted_ts:
        frame_idx = delay_dict_ts[send_ts][1]
        if frame_idx > (last_frame_idx + 1):
            # skipped frames 
            skipped_frames += (frame_idx - last_frame_idx - 1)
            last_ts = sorted_ts[idx_cnt-1]
            print("length", send_ts - last_ts)
            
            missed_tses = np.arange(last_ts, send_ts, 0.1)[1:]
            # print(missed_tses)
            for missed_ts in missed_tses:
                ts.append(missed_ts)
                delay.append(-0.1)
    
        last_frame_idx = frame_idx
        idx_cnt += 1
        ts.append(send_ts)
        delay.append(delay_dict_ts[send_ts][0])
    

    print("skipped frmaes", skipped_frames)

                       
    ts = np.array(ts) - np.min(ts)
    delay = np.array(delay)
    return ts, delay




def get_summary_of_settings(settings):
    setting_summary = open("setting_summary.txt", 'w')
    print("get settings")
    print(len(settings))
    node_num_to_stats = {}
    for setting in settings:
        loc_mean, bw_mean, connect = [], [], []
        setting_summary.write(str(setting) + '\n')
        print("Get stats for setting", setting)
        num_nodes, bw_file, loc, helpee_conf, run_time =\
            int(setting[0]), setting[1], setting[2], setting[3], int(setting[4])
        v2i_bw = get_nodes_v2i_bw(bw_file, run_time, num_nodes, helpee_conf)
        setting_summary.write("-------BW_Summary------\n")
        for i in range(num_nodes):
            setting_summary.write("node%d_bw_mean/std=%f/%f\n"%(i, np.mean(v2i_bw[:, i]), np.std(v2i_bw[:, i])))
        bw_mean.append(np.mean(v2i_bw))
        num_helpees, disconnect_percentage = \
            get_disconect_duration_in_percentage(helpee_conf, run_time, num_nodes)
        connect.append(disconnect_percentage)
        setting_summary.write("------Helpee_Summary------\n")
        setting_summary.write("num_helpees=%d\n"%(num_helpees))
        setting_summary.write("helpee_disconnect_time_percentage=%f\n"%(disconnect_percentage))
        node_dists = get_node_dists(loc)
        setting_summary.write("------Dist_Summay------\n")
        mean_dists = []
        for i in range(num_nodes):
            dists = node_dists[i]
            mean, std = np.mean(dists), np.std(dists)
            mean_dists.append(mean)
            setting_summary.write("node%d_to_other_distances_mean/std=%f/%f\n"%(i,mean, std))
        setting_summary.write("\n")
        loc_mean.append(np.mean(mean_dists))
        if num_nodes not in node_num_to_stats.keys():
            node_num_to_stats[num_nodes] = np.array([loc_mean, bw_mean, connect])
        else:
            node_num_to_stats[num_nodes] = np.hstack((node_num_to_stats[num_nodes], np.array([loc_mean, bw_mean, connect])))
    setting_summary.close()

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(projection='3d')
    print(loc_mean)
    for node_num, stats in node_num_to_stats.items():
        print(node_num)
        ax.scatter(stats[0], stats[1], stats[2], label='%dNodes'%node_num)
    ax.set_xlabel('\nDist mean (m)', linespacing=3.2)
    ax.set_ylabel('\nBW mean (Mbps)', linespacing=3.2)
    ax.set_zlabel('\nDisconnection percentage (%)', linespacing=3.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('configurations.png')

        
def get_control_msg_data_overhead(data_dir, num_nodes):
    node_overheads = []
    for i in range(num_nodes):
        overhead = get_control_msg_data_overhead_per_node(data_dir + '/logs/node%d.log'%i)
        node_overheads.append(overhead)
    print("per node overhead ", node_overheads)
    return np.mean(node_overheads)


def get_control_msg_data_overhead_per_node(filename):
    control_msg_sizes, data_msg_sizes = [], []

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parse = line.split()
            if line.startswith('[route msg]') or line.startswith('[Loc msg'):
                control_msg_sizes.append(int(parse[-2]))
            elif line.startswith('[Sedning Data]'):
                data_msg_sizes.append(int(parse[-2]))
    return np.sum(control_msg_sizes)/np.sum(data_msg_sizes) * 100.0                

        