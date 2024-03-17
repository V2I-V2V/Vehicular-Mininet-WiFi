import sys, os
from numpy.lib.function_base import sort_complex
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import analysis.util as util
import analysis.v2i_bw as v2i_bw, analysis.trajectory as trajectory, analysis.disconnection as disconnection

import run_experiment

plt.rc('font', family='sans-serif', serif='cm10')
plt.rc('text', usetex=False)
plt.rcParams.update({'font.size': 20})

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
    if len(ts) > 0:
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
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    ax.grid(linestyle='--')
    sns.ecdfplot(delay_all, label='Aggregated latency for all nodes')
    # plt.legend()
    
    plt.tight_layout()
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.savefig(dir+'latency-cdf.pdf')


def single_exp_analysis(dir, num_nodes, bw_file, loc_file, helpee_conf, exp_time, config):
    # plot node bw
    v2i_bw.plot_v2i_bw(bw_file, exp_time, num_nodes, dir, helpee_conf)

    # plot trajectory
    trajectory.plot_trajectory(loc_file, dir)

    # plot disconnect trace
    disconnection.plot_disconnect(helpee_conf, exp_time, num_nodes, dir)

        
    # latency_dict, node_to_encode_choices = util.get_stats_on_one_run(dir, num_nodes, helpee_conf, config)
    # np.savetxt(dir+'computation_overhead.txt', latency_dict['sched_latency'])
    # np.savetxt(dir+'control_msg.txt', latency_dict['ctrl-msg-size'])

    # sender_ts_dict = {}
    # sender_adaptive_choice = {}
    # helper_ts_dict = {}
    # receiver_ts_dict = {}
    # receiver_throughput = {}
    # delay_dict = {}
    # delay_dict_ts = {}
    # dl_latency = {}
    # e2e_latency = {}
    # latencies = []
    # for i in range(num_nodes):
    #     receiver_ts_dict[i] = {}
    #     receiver_throughput[i] = []
    # for i in range(num_nodes):
    #     if os.path.exists(dir + 'logs/node%d.log'%i):
    #         sender_ts_dict[i], sender_adaptive_choice[i], encode_time, end_t, summary = util.get_sender_ts(dir + 'logs/node%d.log'%i, config["scheduler"])
    #         dl_latency[i], e2e_latency[i] = summary['dl-latency'], summary['e2e-latency']
    #         latencies += summary['e2e-latency']
    #         helper_ts_dict[i] = get_helper_receive_ts(dir + 'logs/node%d.log'%i)
    # receiver_ts_dict, receiver_thrpt, server_node_dict, _, _ = util.get_receiver_ts(dir + 'logs/server.log')
    # delay_all = np.empty((300,))
    # each_node_delay = {}
    # for i in range(num_nodes):
    #     delay_dict[i], delay_dict_ts[i] = calculate_latency(sender_ts_dict[i], receiver_ts_dict[i])
    #     if i == 0:
    #         delay_all = np.fromiter(delay_dict[i].values(), dtype=float)
    #         each_node_delay[0] = delay_all
    #     else:
    #         delay = np.fromiter(delay_dict[i].values(), dtype=float)
    #         each_node_delay[i] = delay
    #         delay_all = np.concatenate((delay_all, delay))
    
    # print('e2e latency', np.mean(latencies))
    # # Plot variation    
    # # fig = plt.figure(figsize=(5,5))
    # # ax = fig.add_subplot(111)
    # # # ax.set_axisbelow(True)
    # # latency_arr = np.empty((1, len(each_node_delay[0])))
    # # for i in range(num_nodes):
    # #     print(np.array(each_node_delay[i]).reshape(1, -1).shape)
    # #     print(each_node_delay[i])
    # #     latency_arr = np.vstack([latency_arr[:, :200], each_node_delay[i].reshape(1, -1)[:, :200]])
    # #     # sns.ecdfplot(np.fromiter(delay_dict[i].values(), dtype=float), label='node%d'%i)
    # #     # np.savetxt(dir+'node%d_latency.txt'%i, np.fromiter(delay_dict[i].values(), dtype=float))
    # # # plt.xlim([0, 0.5])
    # # latency_arr = latency_arr[1:, :]
    # # mean_latency = np.std(latency_arr, axis=0)[:40]
    # # print("mean", mean_latency)
    # # var_latency = np.std(latency_arr, axis=0)[:40]
    # # min_latency = np.min(latency_arr, axis=0)[:40] + 0.02
    # # max_latency = np.max(latency_arr, axis=0)[:40] + 0.02
    # # np.save('min_lat.npy', min_latency)
    # # np.save('max_lat.npy', max_latency)
    # # # ax.errorbar(np.arange(len(mean_latency)), mean_latency, yerr=var_latency, capsize=2)
    # # # ax.scatter(np.arange(len(mean_latency)), mean_latency, marker='^')
    # # ax.plot(np.arange(len(min_latency)), min_latency, '-o', label='min latency')
    # # ax.plot(np.arange(len(max_latency)), max_latency, '-x', label='max latency')
    # # ax.fill_between(np.arange(len(max_latency)), min_latency, max_latency, color='lime', alpha=0.3)
    # # plt.xlabel("Frame Number")
    # # plt.ylabel("Upload Latency (s)")
    # # plt.tight_layout()
    # # plt.legend()
    # # plt.grid(linestyle='--', axis='y')
    # # # plt.legend()
    # # plt.savefig(dir+'latency-var-each-node.pdf')
    
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111)
    # # sns.ecdfplot(np.std(latency_arr, axis=0))
    # # plt.xlabel("Variance of upload latency (s)")
    # # plt.ylabel("CDF")
    # # plt.tight_layout()
    # # plt.savefig(dir+'latency-var.pdf')

    # fig, axes = plt.subplots(num_nodes, 1, sharex=True, figsize=(6, 3*num_nodes))
    # for i in range(num_nodes):
    #     axes[i].plot(np.arange(len(dl_latency[i])), dl_latency[i], '-x', label='node%d'%i)
    #     axes[i].set_ylabel("Latency (s)")
    #     axes[i].legend()
    #     # axes[i].set_ylim(top=0.3)
    #     np.savetxt(dir+'node_%d_dl_latency.txt'%i, dl_latency[i])
    # plt.tight_layout()
    # plt.xlabel('Frame id')
    # plt.savefig(dir+'dl-latency-each-node.png')

    # fig= plt.figure()
    # ax = fig.add_subplot(111)
    # for i in range(num_nodes):
    #     ax.plot(np.arange(len(e2e_latency[i])), e2e_latency[i], label='node%d'%i)
    # plt.tight_layout()
    # plt.legend()
    # plt.xlabel('Frame id')
    # plt.ylabel('E2E Latency')
    # plt.savefig(dir+'e2e-latency.png')

    # # Plot latency vs. time
    # fig = plt.figure(figsize=(12, 6))
    # ax = fig.add_subplot(111)
    # ax.set_axisbelow(True)
    # for i in range(num_nodes):
    #     # plot time series data
    #     ts, delay = construct_ts_latency_array(delay_dict_ts[i])
    #     if len(ts) > 0:
    #         ax.plot(ts, delay, '--o', label='node%d'%i)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Latency (s)")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(dir+'latency-over-time.png')

    # # plot adaptive encode
    # node_encode_level = {}
    # if len(sender_adaptive_choice[0]) != 0:
    #     bw = np.loadtxt(bw_file)
    #     # fig = plt.figure(figsize=(9, 18))
    #     fig, axes = plt.subplots(num_nodes, 1, sharex=True, figsize=(9, 18))
    #     for i in range(num_nodes):
    #         # ax = fig.add_subplot(num_nodes, 1, i+1)
    #         ax = axes[i]
    #         ax2 = ax.twinx()
    #         ts, delay = construct_ts_latency_array(delay_dict_ts[i])
    #         if len(ts) > 0:
    #             # ax.plot(ts, delay, '--', label='node%d-latency'%i)
    #             encode_levels = []
    #             for k, v in sorted(sender_adaptive_choice[i].items(), key=lambda item: item[0]):
    #                 encode_levels.append(v)
    #             node_encode_level[i] = (ts, encode_levels[:len(ts)])
    #             ax.plot(ts, encode_levels[:len(ts)], label='encode level')
    #             while bw.shape[0] < int(ts[-1]-ts[0]):
    #                 bw = np.vstack((bw, bw[-1]))
                
    #             # ax2.plot(np.arange(int(ts[-1]-ts[0])), bw[:int(ts[-1]-ts[0]), i], label='node%i-bandwidth'%i)
    #             numbers = int(len(ts)/10)

    #             ax2.plot(np.arange(bw[9:numbers+9, i%bw.shape[1]].shape[0]), bw[9:numbers+9, i%bw.shape[1]], label='node%i-bandwidth'%i)
    #             ax.set_xlabel('Time (s)')
    #             ax.set_ylabel('Latency (s)')
    #             ax2.set_ylabel('Bandwidth (Mbps)')
    #             ax.legend(loc='upper left')
    #             ax2.legend()

    #     plt.tight_layout()
    #     plt.savefig(dir+'latency-adaptive-decisions.png')

    # Plot helpees, helpers
    # if len(server_node_dict['time']) > 0:
    #     fig = plt.figure(figsize=(9, 10))
    #     ax = fig.add_subplot(111)
    #     ts = np.array(server_node_dict['time']) - np.min(server_node_dict['time'])
    #     ax.plot(ts, server_node_dict['helper_num'], label='# of helpers')
    #     ax.plot(ts, server_node_dict['helpee_num'], label='# of helpees')
    #     ax.legend()
    #     ax.set_ylabel("Number of helpee, helpers")

    #     ax2 = ax.twinx()
    #     for i in range(num_nodes):
    #         ts_node, delay = construct_ts_latency_array(delay_dict_ts[i])
    #         ts_node = np.array(sorted(delay_dict_ts[i].keys()))-np.min(server_node_dict['time'])
    #         # ts_node += 8 # offset timestamp
    #         ts_combined = np.concatenate((ts_node, ts[ts>ts_node[-1]]))
    #         val_combined = np.concatenate((np.ones(len(ts_node)), np.zeros(len(ts[ts>ts_node[-1]]))))
    #         ax2.plot(ts_combined, val_combined+i/50, '--',  label='node %d frame arrival pattern'%i)
    #     ax2.set_ylabel("Frame Arrival")
    #     ax2.set_yticks([-1, 0, 1, 2])
    #     ax2.legend(loc='lower right')
    #     ax.set_xlabel('Time (s)')
    #     plt.tight_layout()
    #     plt.savefig(dir+'server-helper-helpee-change.png')

    # plot assignments
    # assignment_mode, server_assignments, ts_to_scores, _, _ = util.get_server_assignments(dir + 'logs/server.log')
    # if len(server_assignments) != 0: # proceed if there are assignments
    #     timestamps, assignments, assignment_enums = construct_ts_assignment_array(server_assignments)
    #     print(assignment_enums)
    #     fig = plt.figure(figsize=(9, 12))
    #     ax = fig.add_subplot(511)
    #     ax.plot(timestamps, assignments, color='darkblue', label='assignment')
    #     # ax.set_xlabel('Time (s)')
    #     ax.set_ylabel('Assignment\n(helpee: helper)')
    #     ax.set_yticks(np.arange(0, len(assignment_enums), 1))
    #     ax.set_yticklabels(assignment_enums)
    #     ax.legend()
    #     axes = []
    #     for i in range(3):
    #         axes.append(fig.add_subplot(5,1,i+2))
    #     if len(ts_to_scores) != 0:
    #         print(ts_to_scores)
    #         ts, dist_scores, bw_scores, intf_scores, dist_scores_min, bw_scores_min, intf_scores_min =\
    #              construct_ts_scores_array(ts_to_scores)
    #         axes[0].plot(ts, dist_scores[:0], '--.', label='dist score (harmonic)')
    #         axes[0].plot(ts, dist_scores_min[:0], '--', label='dist score (min)')
    #         axes[1].plot(ts, bw_scores[:0], '--.', label='bw score (harmonic)')
    #         axes[1].plot(ts, bw_scores_min[:0], '--', label='bw score (min)')
    #         axes[2].plot(ts, intf_scores[:0], '--.', label='intf score (harmonic)')
    #         axes[2].plot(ts, intf_scores_min[:0], '--', label='intf score (min)')
    #         # plot the sum of three scores
    #         ax_sum_score = fig.add_subplot(5,1,5)
    #         ax_sum_score.plot(ts, np.array(dist_scores) + np.array(bw_scores) + np.array(intf_scores), '--.',\
    #             label='sum score (harmonic)')
    #         ax_sum_score.plot(ts, np.array(dist_scores_min) + np.array(bw_scores_min) + np.array(intf_scores_min), '--.',\
    #             label='sum score (min)')
    #         ax_sum_score.legend()
    #     for ax in axes:
    #         ax.legend()
    #     ax2.set_xlabel('Time (s)')
    #     # ax2.set_ylabel('Score')
    #     # ax2.legend(loc='lower right')
    #     # ax2.set_yticks(np.arange(-10, 5))
    #     plt.tight_layout()
    #     plt.savefig(dir+'assignments.png')

    # plot_all_delay(dir, delay_all)

    # # plot the summary figure
    # fig = plt.figure(figsize=(9, 7))
    # bws = v2i_bw.get_nodes_v2i_bw(bw_file,exp_time,num_nodes,helpee_conf)
    # cnt = 0
    # selected = [0, 1 ,4]
    # for i in selected:
    #     # plot time series data
    #     ax = fig.add_subplot(len(selected), 2, 2*cnt+1)
    #     if i in latency_dict:
    #         ts, delay = util.construct_ts_latency_array(latency_dict[i])
    #         if len(ts) > 0:
    #             delay = np.array(delay)
    #             if i in [0, 1]:
    #                 ax.plot(ts, delay, '-x', label='helpee node %d'%i)
    #                 ax.set_ylabel("Helpee %d\nLatency (s)"%i)
    #                 if i == 0:
    #                     ax.annotate("V2I Disconnection", fontsize=18, horizontalalignment="center", xy=(12, 0.22), xycoords='data',
    #                         xytext=(19, 0.26), textcoords='data',
    #                         arrowprops=dict(arrowstyle="->, head_width=0.25", connectionstyle="arc3", lw=3)
    #                         )
    #                 else:
    #                     ax.annotate("V2I Disconnection", fontsize=18, horizontalalignment="center", xy=(21.5, 0.16), xycoords='data',
    #                         xytext=(15, 0.20), textcoords='data',
    #                         arrowprops=dict(arrowstyle="->, head_width=0.25", connectionstyle="arc3", lw=3)
    #                         )
    #             else:
    #                 ax.plot(ts, delay, '-o', label='helper')
    #                 ax.set_ylabel("Helper\nLatency (s)")
    #             # ax.legend()
    #             ax.set_ylim(top=0.3)
    #             ax.grid(axis='y', linestyle='--')  
    #             ax.set_xlim(0, exp_time - 20)
    #             ax2 = fig.add_subplot(len(selected), 2, 2*cnt+2)
    #             ax3 = ax2.twinx()
    #             ax2.plot(np.arange(bws[:,i].shape[0]), bws[:,i], '-^', color='blue')
    #             if i in [0, 1]:
    #                 ax2.set_ylabel('Helpee %d\nV2I BW (Mbps)'%i, color='blue')
    #                 if i == 0:
    #                     ax3.annotate("Switch to V2V", fontsize=18, 
    #                                  horizontalalignment="center",
    #                                  xy=(12, 9), xycoords='data',
    #                                  xytext=(16, 11), textcoords='data',
    #                                  arrowprops=dict(arrowstyle="->, head_width=0.3", 
    #                                  connectionstyle="arc3", lw=3)
    #                         )
    #                 else:
    #                     ax3.annotate("Switch to V2V", fontsize=18, 
    #                                  horizontalalignment="center", xy=(22, 9), xycoords='data',
    #                                  xytext=(15, 11), textcoords='data',
    #                                  arrowprops=dict(arrowstyle="->, head_width=0.3",
    #                                  connectionstyle="arc3", lw=3)
    #                         )
    #             else:
    #                 ax2.set_ylabel('Helper\nV2I BW (Mbps)', color='blue')
    #                 ax3.annotate("Help Helpee 0", fontsize=18, 
    #                                  horizontalalignment="center", xy=(13, 9.5), xycoords='data',
    #                                  xytext=(18, 11), textcoords='data',
    #                                  arrowprops=dict(arrowstyle="->, head_width=0.3",
    #                                  connectionstyle="arc3", lw=3)
    #                         )
    #             ax2.tick_params(axis='y', color='blue')
    #             ax2.set_xlim(0, exp_time-20)
    #             ax2.set_ylim(top=20)
    #             ax3.plot(node_encode_level[i][0], node_encode_level[i][1], '--.', label='encode-level', color='darkorange')
    #             ax3.set_yticks([6, 7, 9, 11, 12])
    #             ax3.set_yticklabels(['', 'low', 'medium', 'high', ''])
    #             ax3.set_ylabel('Encoding bitrate', color='darkorange')
    #             ax3.tick_params(axis='y', colors='darkorange')
    #             ax3.set_ylim(6, 12)

    #     cnt += 1
        
    # # if len(server_assignments) != 0:
    # #     ax = fig.add_subplot(num_nodes+1, 2, 2*num_nodes+1)
    # #     ax.plot(timestamps, assignments, color='darkblue', label='assignment')
    # #     ax.set_ylabel('Assignment\n(helpee: helper)')
    # #     ax.set_yticks(np.arange(0, len(assignment_enums), 1))
    # #     ax.set_yticklabels(assignment_enums)
    # ax.set_xlabel("Time (s)")
    # ax2.set_xlabel("Time (s)")  
    # plt.tight_layout()
    # plt.savefig(dir+'summary.pdf')
    
    # plot the summary figure
    # fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9.5, 6))
    # # plot score changes in 1
    # scores = latency_dict['score']
    # tses, ass1_scores, ass2_scores = construct_score_arrays(scores)
    # axes[0].plot(tses, ass1_scores, '-o', label='Assignment 1')
    # axes[0].plot(tses, ass2_scores, '-^', label='Assignment 2')         
    # axes[0].annotate("Assignment Change", fontsize=18, 
    #                 horizontalalignment="center", xy=(38.5, 1.2), xycoords='data',
    #                 xytext=(38, 1.6), textcoords='data',
    #                 arrowprops=dict(arrowstyle="->, head_width=0.2",
    #                 connectionstyle="arc3", lw=2.5)
    #             )
    # axes[0].legend(fontsize=18)
    # axes[0].set_ylabel('Assignment\nScore')
    # ts, delay = construct_ts_latency_array(delay_dict_ts[0])
    # # axes[1].plot(ts, delay, '--.', label='Helpee Node')
    # sender_ts_dict[i], sender_adaptive_choice[i], encode_time, end_t, summary = util.get_sender_ts(dir + 'logs/node0.log', config["scheduler"])
    # axes[1].plot(np.arange(0, int(len(summary['e2e-raw'])/10), 0.1), summary['e2e-raw'][:int(len(summary['e2e-raw'])/10)*10], '--.', label='Helpee Node')
    # axes[1].axhline(y=0.5, linestyle='--', color='r')
    # axes[1].set_ylabel('Detection\nLatency (s)')
    # plt.xlim(20, 45)
    # plt.xlabel('Time (s)')
    # plt.legend(fontsize=18)
    # plt.tight_layout()
    # plt.savefig(dir+'timeline.pdf')
      


def construct_score_arrays(scores):
    tses, ass1_scores, ass2_scores = [], [], []
    for tup in sorted(scores):
        ts, ass1, ass2 = tup[0], tup[1], tup[2]
        tses.append(ts)
        ass1_scores.append(ass1)
        ass2_scores.append(ass2)
    tses = np.array(tses) - np.min(tses) + 0.4
    return tses, ass1_scores, ass2_scores

if __name__ == '__main__':
    dir=sys.argv[1]
    config_params = run_experiment.parse_config_from_file(dir + "/config.txt")
    num_nodes = int(config_params['num_of_nodes'])
    bw_file = config_params['network_trace']
    loc_file = config_params['location_file']
    helpee_conf = config_params['helpee_conf']
    exp_time = int(config_params['t'])
    single_exp_analysis(dir, num_nodes, bw_file, loc_file, helpee_conf, exp_time, config_params)
