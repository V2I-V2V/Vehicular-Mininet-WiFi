import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import analysis.util as util
import analysis.v2i_bw as v2i_bw, analysis.trajectory as trajectory, analysis.disconnection as disconnection

font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)

MAX_FRAMES = 80

np.set_printoptions(precision=3)



def get_helper_receive_ts(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        helper_receive_ts = {}
        for line in lines:
            if line.startswith("[Received a full frame/oxts]"):
                parse = line.split()
                f_id, sender_id, ts = int(parse[6]), int(parse[-2]), float(parse[-1])
                if sender_id not in helper_receive_ts.keys():
                    helper_receive_ts[sender_id] = [ts]
                else:
                    helper_receive_ts[sender_id].append(ts)
        f.close()
        return helper_receive_ts


def calculate_latency(sender_ts_dict, receiver_ts_dict):
    delay_dict, delay_dict_ts = {}, {}
    for k, v in receiver_ts_dict.items():
        delay_dict[k] = v - sender_ts_dict[k]
        delay_dict_ts[sender_ts_dict[k]] = v - sender_ts_dict[k]
    return delay_dict, delay_dict_ts
    

def construct_ts_latency_array(delay_dict_ts):
    ts, delay = [], []
    for i in sorted(delay_dict_ts.keys()):
        ts.append(i)
        delay.append(delay_dict_ts[i])
    ts = np.array(ts) - np.min(ts)
    delay = np.array(delay)
    return ts, delay


def construct_ts_assignment_array(server_assignments):
    ts, assignments, assignment_enums = [], [], []
    for val in server_assignments.values():
        if val not in assignment_enums:
            assignment_enums.append(val)
    for timestamp in sorted(server_assignments.keys()):
        ts.append(timestamp)
        assignments.append(assignment_enums.index(server_assignments[timestamp]))
    ts = np.array(ts) - np.min(ts)
    assignments = np.array(assignments)
    return ts, assignments, assignment_enums


def construct_ts_scores_array(scores):
    ts, dist_scores, bw_scores, intf_scores = [], [], [], []
    for timestamp, score in scores.items():
        ts.append(timestamp)
        dist_scores.append(score[0])
        bw_scores.append(score[1])
        intf_scores.append(score[2])
    ts = np.array(ts) - np.min(ts)
    return ts, dist_scores, bw_scores, intf_scores


def single_exp_analysis(dir, num_nodes, bw_file, loc_file, helpee_conf, exp_time):
    # plot node bw
    v2i_bw.plot_v2i_bw(bw_file, exp_time, num_nodes, dir)

    # plot trajectory
    trajectory.plot_trajectory(loc_file, dir)

    # plot disconnect trace
    disconnection.plot_disconnect(helpee_conf, exp_time, num_nodes, dir)


    sender_ts_dict = {}
    sender_adaptive_choice = {}
    helper_ts_dict = {}
    receiver_ts_dict = {}
    receiver_throughput = {}
    delay_dict = {}
    delay_dict_ts = {}
    for i in range(num_nodes):
        receiver_ts_dict[i] = {}
        receiver_throughput[i] = []
    for i in range(num_nodes):
        sender_ts_dict[i], sender_adaptive_choice[i], encode_time = util.get_sender_ts(dir + 'logs/node%d.log'%i)
        helper_ts_dict[i] = get_helper_receive_ts(dir + 'logs/node%d.log'%i)
    receiver_ts_dict, receiver_thrpt, server_node_dict = util.get_receiver_ts(dir + 'logs/server.log')
    delay_all = np.empty((300,))
    for i in range(num_nodes):
        delay_dict[i], delay_dict_ts[i] = calculate_latency(sender_ts_dict[i], receiver_ts_dict[i])
        # print(len(delay_dict[i]))
        if i == 0:
            delay_all = np.fromiter(delay_dict[i].values(), dtype=float)
            # print(delay_all)
        else:
            delay = np.fromiter(delay_dict[i].values(), dtype=float)
            delay_all = np.concatenate((delay_all, delay))

    # Plot distribution    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    for i in range(num_nodes):
        sns.ecdfplot(np.fromiter(delay_dict[i].values(), dtype=float), label='node%d'%i)
    # plt.xlim([0, 0.5])
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig(dir+'latency-cdf-each-node.png')

    # Plot latency vs. time
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    for i in range(num_nodes):
        # plot time series data
        ts, delay = construct_ts_latency_array(delay_dict_ts[i])
        ax.plot(ts, delay, '--o', label='node%d'%i)
        np.savetxt(dir+'node%d_delay.txt'%i, np.fromiter(delay_dict[i].values(), dtype=float))
        np.savetxt(dir+'node%d_thrpt.txt'%i, np.array(receiver_thrpt[i]))
    plt.xlabel("Time (s)")
    plt.ylabel("Latency (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir+'latency-over-time.png')

    fig = plt.figure(figsize=(9, 12))
    for i in range(num_nodes):
        # plot time series data
        ax = fig.add_subplot(num_nodes,1, i+1)
        ts, delay = construct_ts_latency_array(delay_dict_ts[i])
        ax.plot(ts, delay, '--o', label='node%d'%i)
        ax.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Latency (s)")
    plt.tight_layout()
    plt.savefig(dir+'latency-over-time-each-node.png')

    # plot adaptive encode
    if len(sender_adaptive_choice[0]) != 0:
        bw = np.loadtxt(bw_file)
        fig = plt.figure(figsize=(9, 7))
        for i in range(num_nodes):
            ax = fig.add_subplot(num_nodes, 1, i+1)
            ax2 = ax.twinx()
            ts, delay = construct_ts_latency_array(delay_dict_ts[i])
            ax.plot(ts, delay, '--', label='node%d-latency'%i)
            encode_levels = []
            for k, v in sorted(sender_adaptive_choice[i].items(), key=lambda item: item[0]):
                encode_levels.append(v)
            ax.plot(ts, encode_levels, label='encode level')
            while bw.shape[0] < int(ts[-1]-ts[0]):
                bw = np.vstack((bw, bw[-1]))
            ax2.plot(np.arange(int(ts[-1]-ts[0])), bw[:int(ts[-1]-ts[0]), i], label='node%i-bandwidth'%i)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Latency (s)')
            ax2.set_ylabel('Bandwidth (Mbps)')
            # ax2.set_ylim(-4, 22)
            # ax.set_ylim(0, 5)
            ax.legend(loc='upper left')
            ax2.legend()

        plt.tight_layout()
        plt.savefig(dir+'latency-adaptive.png')

    # Plot helpees, helpers
    fig = plt.figure(figsize=(9, 10))
    ax = fig.add_subplot(111)
    ts = np.array(server_node_dict['time']) - np.min(server_node_dict['time'])
    ax.plot(ts, server_node_dict['helper_num'], label='# of helpers')
    ax.plot(ts, server_node_dict['helpee_num'], label='# of helpees')
    ax.legend()
    ax.set_ylabel("Number of helpee, helpers")

    ax2 = ax.twinx()
    for i in range(num_nodes):
        ts_node, delay = construct_ts_latency_array(delay_dict_ts[i])
        ts_node = np.array(sorted(delay_dict_ts[i].keys()))-np.min(server_node_dict['time'])
        # ts_node += 8 # offset timestamp
        ts_combined = np.concatenate((ts_node, ts[ts>ts_node[-1]]))
        val_combined = np.concatenate((np.ones(len(ts_node)), np.zeros(len(ts[ts>ts_node[-1]]))))
        ax2.plot(ts_combined, val_combined+i/50, '--',  label='node %d frame arrival pattern'%i)
        # ax2.scatter(ts_combined, val_combined+i/50,  label='node %d frame arrival pattern'%i)
    # ax2.plot(ts[ts>ts_node[-1]], np.zeros(len(ts[ts>ts_node[-1]])))
    ax2.set_ylabel("Frame Arrival")
    ax2.set_yticks([-1, 0, 1, 2])
    ax2.legend(loc='lower right')
    ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(dir+'server-helper-helpee-change.png')

    # plot assignments
    server_assignments, ts_to_scores = util.get_server_assignments(dir + 'logs/server.log')
    if len(server_assignments) != 0: # proceed if there are assignments
        timestamps, assignments, assignment_enums = construct_ts_assignment_array(server_assignments)
        print(assignment_enums)
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(timestamps, assignments, color='darkblue', label='assignment')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Assignment (helpee: helper)')
        ax.set_yticks(np.arange(0, len(assignment_enums), 1))
        ax.set_yticklabels(assignment_enums)
        ax.legend()
        if len(ts_to_scores) != 0:
            ts, dist_scores, bw_scores, intf_scores = construct_ts_scores_array(ts_to_scores)
            ax2 = ax.twinx()
            ax2.plot(ts, dist_scores, '--', label='dist score')
            ax2.plot(ts, bw_scores, '--', label='bw score')
            ax2.plot(ts, intf_scores, '--', label='intf score')
        ax2.set_ylabel('Score')
        ax2.legend(loc='lower right')
        ax2.set_yticks(np.arange(-10, 5))
        plt.tight_layout()
        plt.savefig(dir+'assignments.png')

    # plot overall delay
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    sns.ecdfplot(delay_all)
    # plt.xlim([0, 0.5])
    np.savetxt(dir+'all_delay.txt', delay_all)
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    # plt.legend()
    plt.savefig(dir+'latency-cdf.png')


if __name__ == '__main__':
    dir=sys.argv[1]
    num_nodes = int(sys.argv[2])
    bw_file = sys.argv[3]
    loc_file = sys.argv[4]
    helpee_conf = sys.argv[5]
    exp_time = int(sys.argv[6])
    single_exp_analysis(dir, num_nodes, bw_file, loc_file, helpee_conf, exp_time)
