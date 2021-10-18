import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import analysis.util as util
import analysis.v2i_bw as v2i_bw, analysis.trajectory as trajectory, analysis.disconnection as disconnection

import run_experiment

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
    ts, dist_scores, bw_scores, intf_scores, dist_scores_min, bw_scores_min, intf_scores_min  = [], [], [], [], [], [], []
    for timestamp, score in scores.items():
        ts.append(timestamp)
        dist_scores.append([score[0], score[1]])
        bw_scores.append([score[2], score[3]])
        intf_scores.append([score[4], score[5]])
        dist_scores_min.append([score[6], score[7]])
        bw_scores_min.append([score[8], score[9]])
        intf_scores_min.append([score[10], score[11]])
    ts = np.array(ts) - np.min(ts)
    return ts, np.array(dist_scores), np.array(bw_scores), np.array(intf_scores),\
         np.array(dist_scores_min), np.array(bw_scores_min), np.array(intf_scores_min)


def plot_all_delay(dir, delay_all):
    # plot overall delay
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    sns.ecdfplot(delay_all)
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.savefig(dir+'latency-cdf.png')


def single_exp_analysis(dir, num_nodes, bw_file, loc_file, helpee_conf, exp_time):
    # plot node bw
    v2i_bw.plot_v2i_bw(bw_file, exp_time, num_nodes, dir, helpee_conf)

    # plot trajectory
    trajectory.plot_trajectory(loc_file, dir)

    # plot disconnect trace
    disconnection.plot_disconnect(helpee_conf, exp_time, num_nodes, dir)

        
    latency_dict, node_to_encode_choices = util.get_stats_on_one_run(dir, num_nodes, helpee_conf)

    sender_ts_dict = {}
    sender_adaptive_choice = {}
    helper_ts_dict = {}
    receiver_ts_dict = {}
    receiver_throughput = {}
    delay_dict = {}
    delay_dict_ts = {}
    dl_latency = {}
    e2e_latency = {}
    for i in range(num_nodes):
        receiver_ts_dict[i] = {}
        receiver_throughput[i] = []
    for i in range(num_nodes):
        sender_ts_dict[i], sender_adaptive_choice[i], encode_time, end_t, summary = util.get_sender_ts(dir + 'logs/node%d.log'%i)
        dl_latency[i], e2e_latency[i] = summary['dl-latency'], summary['e2e-latency']
        helper_ts_dict[i] = get_helper_receive_ts(dir + 'logs/node%d.log'%i)
    receiver_ts_dict, receiver_thrpt, server_node_dict, _, _ = util.get_receiver_ts(dir + 'logs/server.log')
    delay_all = np.empty((300,))
    for i in range(num_nodes):
        delay_dict[i], delay_dict_ts[i] = calculate_latency(sender_ts_dict[i], receiver_ts_dict[i])
        if i == 0:
            delay_all = np.fromiter(delay_dict[i].values(), dtype=float)
        else:
            delay = np.fromiter(delay_dict[i].values(), dtype=float)
            delay_all = np.concatenate((delay_all, delay))

    # Plot distribution    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    for i in range(num_nodes):
        sns.ecdfplot(np.fromiter(delay_dict[i].values(), dtype=float), label='node%d'%i)
        np.savetxt(dir+'node%d_latency.txt'%i, np.fromiter(delay_dict[i].values(), dtype=float))
    # plt.xlim([0, 0.5])
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig(dir+'latency-cdf-each-node.png')

    fig, axes = plt.subplots(num_nodes, 1, sharex=True, figsize=(6, 3*num_nodes))
    for i in range(num_nodes):
        axes[i].plot(np.arange(len(dl_latency[i])), dl_latency[i], '-x', label='node%d'%i)
        axes[i].set_ylabel("Latency (s)")
        axes[i].legend()
        axes[i].set_ylim(top=0.3)
        np.savetxt(dir+'node_%d_dl_latency.txt'%i, dl_latency[i])
    plt.tight_layout()
    plt.xlabel('Frame id')
    plt.savefig(dir+'dl-latency-each-node.png')

    fig= plt.figure()
    ax = fig.add_subplot(111)
    for i in range(num_nodes):
        ax.plot(np.arange(len(e2e_latency[i])), e2e_latency[i], label='node%d'%i)
    plt.tight_layout()
    plt.legend()
    plt.xlabel('Frame id')
    plt.ylabel('E2E Latency')
    plt.savefig(dir+'e2e-latency.png')

    # Plot latency vs. time
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    for i in range(num_nodes):
        # plot time series data
        ts, delay = construct_ts_latency_array(delay_dict_ts[i])
        ax.plot(ts, delay, '--o', label='node%d'%i)
    plt.xlabel("Time (s)")
    plt.ylabel("Latency (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir+'latency-over-time.png')

    # plot adaptive encode
    if len(sender_adaptive_choice[0]) != 0:
        bw = np.loadtxt(bw_file)
        # fig = plt.figure(figsize=(9, 18))
        fig, axes = plt.subplots(num_nodes, 1, sharex=True, figsize=(9, 18))
        for i in range(num_nodes):
            # ax = fig.add_subplot(num_nodes, 1, i+1)
            ax = axes[i]
            ax2 = ax.twinx()
            ts, delay = construct_ts_latency_array(delay_dict_ts[i])
            ax.plot(ts, delay, '--', label='node%d-latency'%i)
            encode_levels = []
            for k, v in sorted(sender_adaptive_choice[i].items(), key=lambda item: item[0]):
                encode_levels.append(v)
            ax.plot(ts, encode_levels[:len(ts)], label='encode level')
            while bw.shape[0] < int(ts[-1]-ts[0]):
                bw = np.vstack((bw, bw[-1]))
            ax2.plot(np.arange(int(ts[-1]-ts[0])), bw[:int(ts[-1]-ts[0]), i], label='node%i-bandwidth'%i)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Latency (s)')
            ax2.set_ylabel('Bandwidth (Mbps)')
            ax.legend(loc='upper left')
            ax2.legend()

        plt.tight_layout()
        plt.savefig(dir+'latency-adaptive-decisions.png')

    # Plot helpees, helpers
    if len(server_node_dict['time']) > 0:
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
        ax2.set_ylabel("Frame Arrival")
        ax2.set_yticks([-1, 0, 1, 2])
        ax2.legend(loc='lower right')
        ax.set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(dir+'server-helper-helpee-change.png')

    # plot assignments
    assignment_mode, server_assignments, ts_to_scores, _, _ = util.get_server_assignments(dir + 'logs/server.log')
    if len(server_assignments) != 0: # proceed if there are assignments
        timestamps, assignments, assignment_enums = construct_ts_assignment_array(server_assignments)
        print(assignment_enums)
        fig = plt.figure(figsize=(9, 12))
        ax = fig.add_subplot(511)
        ax.plot(timestamps, assignments, color='darkblue', label='assignment')
        # ax.set_xlabel('Time (s)')
        ax.set_ylabel('Assignment\n(helpee: helper)')
        ax.set_yticks(np.arange(0, len(assignment_enums), 1))
        ax.set_yticklabels(assignment_enums)
        ax.legend()
        axes = []
        for i in range(3):
            axes.append(fig.add_subplot(5,1,i+2))
        if len(ts_to_scores) != 0:
            print(ts_to_scores)
            ts, dist_scores, bw_scores, intf_scores, dist_scores_min, bw_scores_min, intf_scores_min =\
                 construct_ts_scores_array(ts_to_scores)
            axes[0].plot(ts, dist_scores[:0], '--.', label='dist score (harmonic)')
            axes[0].plot(ts, dist_scores_min[:0], '--', label='dist score (min)')
            axes[1].plot(ts, bw_scores[:0], '--.', label='bw score (harmonic)')
            axes[1].plot(ts, bw_scores_min[:0], '--', label='bw score (min)')
            axes[2].plot(ts, intf_scores[:0], '--.', label='intf score (harmonic)')
            axes[2].plot(ts, intf_scores_min[:0], '--', label='intf score (min)')
            # plot the sum of three scores
            ax_sum_score = fig.add_subplot(5,1,5)
            ax_sum_score.plot(ts, np.array(dist_scores) + np.array(bw_scores) + np.array(intf_scores), '--.',\
                label='sum score (harmonic)')
            ax_sum_score.plot(ts, np.array(dist_scores_min) + np.array(bw_scores_min) + np.array(intf_scores_min), '--.',\
                label='sum score (min)')
            ax_sum_score.legend()
        for ax in axes:
            ax.legend()
        ax2.set_xlabel('Time (s)')
        # ax2.set_ylabel('Score')
        # ax2.legend(loc='lower right')
        # ax2.set_yticks(np.arange(-10, 5))
        plt.tight_layout()
        plt.savefig(dir+'assignments.png')

    plot_all_delay(dir, delay_all)

    # plot the summary figure
    fig = plt.figure(figsize=(18, 12))
    bws = v2i_bw.get_nodes_v2i_bw(bw_file,exp_time,num_nodes,helpee_conf)
    for i in range(num_nodes):
        # plot time series data
        ax = fig.add_subplot(num_nodes+1, 2, 2*i+1)
        # ts, delay = construct_ts_latency_array(delay_dict_ts[i])
        ts, delay = util.construct_ts_latency_array(latency_dict[i])
        delay = np.array(delay)
        # print(np.argwhere(delay < 0))
        ax.plot(ts, delay, '--o', label='node%d'%i)
        ax.legend()
        ax.set_ylabel("Latency (s)")
        ax.set_xlim(0, exp_time-14)
        ax2 = fig.add_subplot(num_nodes+1, 2, 2*i+2)
        ax2.plot(np.arange(bws[:,i].shape[0]), bws[:,i])
        ax2.set_ylabel('BW')
        ax2.set_ylim(top=50)

    if len(server_assignments) != 0:
        ax = fig.add_subplot(num_nodes+1, 2, 2*num_nodes+1)
        ax.plot(timestamps, assignments, color='darkblue', label='assignment')
        ax.set_ylabel('Assignment\n(helpee: helper)')
        ax.set_yticks(np.arange(0, len(assignment_enums), 1))
        ax.set_yticklabels(assignment_enums)

    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(dir+'summary.png')


if __name__ == '__main__':
    dir=sys.argv[1]
    config_params = run_experiment.parse_config_from_file(dir + "/config.txt")
    num_nodes = int(config_params['num_of_nodes'])
    bw_file = config_params['network_trace']
    loc_file = config_params['location_file']
    helpee_conf = config_params['helpee_conf']
    exp_time = int(config_params['t'])
    single_exp_analysis(dir, num_nodes, bw_file, loc_file, helpee_conf, exp_time)
