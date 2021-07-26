import numpy as np
import math

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
                
    return sender_ts, encode_choice, encode_t


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


def get_helpees(helpee_conf):
    conf = np.loadtxt(helpee_conf)
    if len(conf) == 0:
        return np.array([])
    elif conf.ndim == 1:
        return np.array([conf[0]])
    else:
        return conf[0]
    

def get_stats_on_one_run(dir, num_nodes, helpee_conf):
    helpees = get_helpees(helpee_conf)
    sender_ts_dict, encode_choice_dict = {}, {}
    latency_dict = {}
    for i in range(num_nodes):
        sender_ts_dict[i], encode_choice_dict[i], encode_t = get_sender_ts(dir + '/logs/node%d.log'%i)
        latency_dict[i] = {}
    receiver_ts_dict, receiver_thrpt, server_helper_dict = get_receiver_ts(dir + '/logs/server.log')
    # calculate delay
    all_delay = []
    helpee_delay = []
    helper_delay = []
    for i in range(num_nodes):
        for frame_idx, recv_ts in receiver_ts_dict[i].items():
            send_ts = sender_ts_dict[i][frame_idx]
            latency = recv_ts-sender_ts_dict[i][frame_idx]
            latency_dict[i][send_ts] = [latency, frame_idx] # add adptation choice
            all_delay.append(latency)
            if i in helpees:
                helpee_delay.append(latency)
            else:
                helper_delay.append(latency)
    latency_dict['all'] = np.array(all_delay)
    latency_dict['helpee'] = np.array(helpee_delay)
    latency_dict['helper'] = np.array(helper_delay)
    return latency_dict