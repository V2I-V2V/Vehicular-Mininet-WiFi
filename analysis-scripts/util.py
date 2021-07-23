import numpy as np

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
            elif line.startswith("latency:"):
                num_chunks = int(parse[-1])
                encode_choice[frame] = num_chunks
                
    return sender_ts, encode_choice


def get_receiver_ts(filename):
    receiver_throughput = {}
    receiver_ts_dict = {}
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
        f.close()
    return receiver_ts_dict, receiver_throughput


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
        sender_ts_dict[i], encode_choice_dict[i] = get_sender_ts(dir + '/logs/node%d.log'%i)
        latency_dict[i] = {}
    receiver_ts_dict, receiver_thrpt = get_receiver_ts(dir + '/logs/server.log')
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