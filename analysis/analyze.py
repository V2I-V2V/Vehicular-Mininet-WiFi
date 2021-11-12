import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_experiment

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.patches import Rectangle
import matplotlib.legend as mlegend
from util import *
from analyze_single_exp import construct_ts_assignment_array, construct_ts_scores_array

font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)

setting_to_folder = {}
result_each_run = {}
result_per_node = {}
num_nodes = 6
LATENCY_THRESHOLD = 0.2
SSIM_THRESHOLD = None

SCHEDULERS = []
LOC = []
BW = []
HELPEE = []
config_set = set()

def tablelegend(ax, col_labels=None, row_labels=None, title_label="", *args, **kwargs):
    """
    Place a table legend on the axes.
    
    Creates a legend where the labels are not directly placed with the artists, 
    but are used as row and column headers, looking like this:
    
    title_label   | col_labels[1] | col_labels[2] | col_labels[3]
    -------------------------------------------------------------
    row_labels[1] |
    row_labels[2] |              <artists go there>
    row_labels[3] |
    
    
    Parameters
    ----------
    
    ax : `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.
        
    col_labels : list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.
        
    row_labels : list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.
        
    title_label : str, optional
        Label for the top left corner in the legend table.
        
    ncol : int
        Number of columns.
        

    Other Parameters
    ----------------
    
    Refer to `matplotlib.legend.Legend` for other parameters.
    
    """
    #################### same as `matplotlib.axes.Axes.legend` #####################
    handles, labels, extra_args, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)
    if len(extra_args):
        raise TypeError('legend only accepts two non-keyword arguments')
    
    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
    #################### modifications for table legend ############################
    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -2)
        title_label = [title_label]
        
        # blank rectangle handle
        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]
        
        # empty label
        empty = [""]
        
        # number of rows infered from number of handles and desired number of columns
        nrow = len(handles) // ncol
        
        # organise the list of handles and labels for table construction
        if col_labels is None:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels  = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = []
            leg_labels  = []
        else:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels  = title_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels  += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels  += empty * nrow
        
        # Create legend
        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels, ncol=ncol+int(row_labels is not None), handletextpad=handletextpad, fontsize=13, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_


def get_key_from_config(config, dir=''):
    global num_nodes
    scheduler = config["scheduler"]
    network = config["network_trace"].split('/')[-1][:-4]
    mobility = config["location_file"].split('/')[-1][:-4]
    helpee = config["helpee_conf"].split('/')[-1][:-4]
    num_nodes = int(config["num_of_nodes"])
    if "adapt_frame_skipping" in config.keys():
        adapt_frame_skipping = int(config["adapt_frame_skipping"])
    else:
        adapt_frame_skipping = 0

    if "adaptive_encode" in config.keys():
        adapt_encoding = int(config["adaptive_encode"])
    else:
        adapt_encoding = 0
    if adapt_frame_skipping == 1 or adapt_encoding == 1:
        scheduler += '-adapt'
    # if 'combine_method' in config.keys() and scheduler == 'combined':
    #     scheduler += ('-' + config['combine_method'])
    if 'add_loc_noise' in config.keys() and config['add_loc_noise'] == '1':
        scheduler += '-loc'
    # if 'score_method' in config.keys() and scheduler == 'combined-op_sum':
    #     scheduler += ('-' + config['score_method'])

    if "adaptive_encode" in config.keys():
        adaptive = config["adaptive_encode"]
    else:
        adaptive = "0" 
    if scheduler not in SCHEDULERS: # TODO: Use a set
        SCHEDULERS.append(scheduler)
    if network not in BW:
        BW.append(network)
    if mobility not in LOC:
        LOC.append(mobility)
    if helpee not in HELPEE:
        HELPEE.append(helpee)
    if "multi" in config.keys():
        is_multi = config["multi"]
        conf_key = (dir, scheduler, network, mobility, is_multi, helpee, adaptive, adapt_frame_skipping)
    else:
        conf_key = (dir, scheduler, network, mobility, helpee, adaptive, adapt_frame_skipping)
    return conf_key


def compare_two_sched(data_folder1, data_folder2):
    ass1_mode, server_ass1, ts_to_scores1, score_combined, score_min = get_server_assignments(data_folder1+'/logs/server.log') 
    if ass1_mode == 'distributed':
        ass1_mode, server_ass1 = get_distributed_helper_assignments(data_folder1, num_nodes)
    ass2_mode, server_ass2, ts_to_scores2, _, _ = get_server_assignments(data_folder2+'/logs/server.log')



    if ass2_mode == 'distributed':
        ass2_mode, server_ass2 = get_distributed_helper_assignments(data_folder2, num_nodes)
    config1 = run_experiment.parse_config_from_file(data_folder1 + '/config.txt')
    config2 = run_experiment.parse_config_from_file(data_folder2 + '/config.txt')
    sched1_key, sched2_key = get_key_from_config(config1, data_folder1), get_key_from_config(config2, data_folder2)
    ass1_rst, ass2_rst = result_each_run[sched1_key], result_each_run[sched2_key]

    fig = plt.figure(figsize=(18,15))
    # total_num_subfigures = (num_nodes + 1) * 2
    sched1_ax_list, sched2_ax_list = [], []
    n_nodes = num_nodes - 3 
    for i in range(n_nodes): # num_nodes
        if i == 0:
            ax1 = fig.add_subplot(n_nodes+1, 2, 2*i+1)
            ax2 = fig.add_subplot(n_nodes+1, 2, 2*i+2)
        else:
            ax1 = fig.add_subplot(n_nodes+1, 2, 2*i+1, sharex=ax1)
            ax2 = fig.add_subplot(n_nodes+1, 2, 2*i+2, sharex=ax2)
        s1_ts, s1_latency = construct_ts_latency_array(ass1_rst[i])
        ax1.plot(s1_ts, s1_latency, color=sched_to_color[config1['scheduler']], label='node%d'%i)
        ax1.set_ylabel('Latency (s)')
        s2_ts, s2_latency = construct_ts_latency_array(ass2_rst[i])
        ax2.plot(s2_ts, s2_latency, color=sched_to_color[config2['scheduler']], label='node%d'%i)
        ax2.set_ylabel('Latency (s)')
        ax1.legend()
        ax2.legend()
        ax1.set_xlim([10, 15])
        ax2.set_xlim([10, 15])
        sched1_ax_list.append(ax1)
        sched2_ax_list.append(ax2)

    # plot assignments
    ax = fig.add_subplot(n_nodes+1, 2, 2*n_nodes+1)
    timestamps, assignments, assignment_enums = construct_ts_assignment_array(server_ass1)
    ax.plot(timestamps, assignments, color=sched_to_color[config1['scheduler']], label='assignment')
    ax.set_ylabel('Assignment\n(helpee: helper)')
    ax.set_xlabel(sched1_key[1], fontsize=30)
    ax.set_yticks(np.arange(0, len(assignment_enums), 1))
    ax.set_yticklabels(assignment_enums)
    sched1_ax_list.append(ax)

    ax2 = fig.add_subplot(n_nodes+1, 2, 2*n_nodes+2)
    timestamps, assignments, assignment_enums = construct_ts_assignment_array(server_ass2)
    ax2.plot(timestamps, assignments, color=sched_to_color[config2['scheduler']], label='assignment')
    ax2.set_ylabel('Assignment\n(helpee: helper)')
    ax2.set_xlabel(sched2_key[1], fontsize=30)
    ax2.set_yticks(np.arange(0, len(assignment_enums), 1))
    ax2.set_yticklabels(assignment_enums)
    sched2_ax_list.append(ax2)

    ax.set_xlim([10, 15])
    ax2.set_xlim([10, 15])
    # ax.set_xlabel('Time (s)')
    # ax2.set_xlabel('Time (s)')

    sched1_ax_list[0].get_shared_x_axes().join(*sched2_ax_list)
    sched2_ax_list[0].get_shared_x_axes().join(*sched2_ax_list)

    plt.tight_layout()
    plt.savefig('analysis-results/compare-ass.png')

    # timestamps, assignments, assignment_enums = construct_ts_assignment_array(server_ass1)
    # timestamps2, assignments2, assignment_enums2 = construct_ts_assignment_array(server_ass2)
    # fig, axes = plt.subplots(2, 5, sharex=True, figsize=(72,9))
    # print(score_combined['comb'])
    # axes[0, 0].plot(np.linspace(0, len(score_combined['min'])*0.2, len(score_combined['min'])), score_combined['comb'], label='combined')
    # axes[0, 0].plot(np.linspace(0, len(score_combined['min'])*0.2, len(score_combined['min'])), score_combined['min'], '--', label='assignment score\nchose by combined-min')
    # axes[0, 0].axvline(1.4579229354858398, linestyle='-.', color='r', alpha=0.7, label='assignment change')
    # # axes[0, 0].set_xticks(np.linspace(0, len(score_combined['min'])*0.2, 0.2))
    # axes[0, 0].legend()
    # axes[0, 0].set_ylabel('Combiend sched\nscores')
    # axes[1, 0].plot(np.linspace(0, len(score_min['min'])*0.2, len(score_min['min'])), score_min['min'], label='combined-min')
    # axes[1, 0].plot(np.linspace(0, len(score_min['comb'])*0.2, len(score_min['comb'])), score_min['comb'], '--', label='assignment score\nchose by combined')
    # # axes[1, 0].set_xticks(np.linspace(0, int(len(score_min['min'])*0.2), 0.2))
    # axes[1, 0].axvline(1.4579229354858398, linestyle='-.', color='r', alpha=0.7, label='assignment change')
    # axes[1, 0].legend()
    # axes[1, 0].set_ylabel('Combiend-min sched\nscores')
    # axes[0, 1].plot(timestamps, assignments, color=sched_to_color[config1['scheduler']], label=sched1_key[1])
    # axes[0, 1].set_ylabel('Assignment\n(helpee: helper)')
    # axes[0, 1].set_yticks(np.arange(0, len(assignment_enums), 1))
    # axes[0, 1].set_yticklabels(assignment_enums)
    # axes[0, 1].legend()
    # axes[1, 1].plot(timestamps2, assignments2, color=sched_to_color[config2['scheduler']], label=sched2_key[1])
    # print(timestamps[7])
    # axes[1, 1].set_ylabel('Assignment\n(helpee: helper)')
    # axes[1, 1].set_yticks(np.arange(0, len(assignment_enums2), 1))
    # axes[1, 1].set_yticklabels(assignment_enums2)
    # axes[1, 1].legend()

    # ts1, dist_scores1, bw_scores1, intf_scores1, dist_scores_min1, bw_scores_min1, intf_scores_min1 =\
    #             construct_ts_scores_array(ts_to_scores1)
    # # ts2, dist_scores2, bw_scores2, intf_scores2, dist_scores_min2, bw_scores_min2, intf_scores_min2 =\
    # #             construct_ts_scores_array(ts_to_scores2)

    # # axes[0, 2].plot(ts2, dist_scores2[:, 0], label='combined')
    # # axes[0, 2].plot(ts2, dist_scores2[:, 1], label='combined')
    # # axes[0, 2].plot(ts1, dist_scores1[:, 0], '--', label='assignment score\nchose by combined-min')
    # # axes[0, 2].plot(ts1, dist_scores1[:, 1], '--', label='assignment score\nchose by combined-min')
    # # axes[0, 2].axvline(1.4579229354858398, linestyle='-.', color='r', alpha=0.7, label='assignment change')
    # # axes[0, 2].legend()
    # # axes[0, 3].plot(ts2, bw_scores2, label='combined')
    # # axes[0, 3].plot(ts1, bw_scores1, '--', label='assignment score\nchose by combined-min')
    # # axes[0, 3].axvline(1.4579229354858398, linestyle='-.', color='r', alpha=0.7, label='assignment change')
    # # axes[0, 3].legend()
    # # axes[0, 4].plot(ts2, intf_scores2, label='combined')
    # # axes[0, 4].plot(ts1, intf_scores1, '--', label='assignment score\nchose by combined-min')
    # # axes[0, 4].axvline(1.4579229354858398, linestyle='-.', color='r', alpha=0.7, label='assignment change')
    # # axes[0, 4].legend()

    # axes[1, 2].plot(ts1, dist_scores_min1[:, 0], label='combined-min')
    # axes[1, 2].plot(ts1, dist_scores_min1[:, 1], label='combined-min')
    # axes[1, 2].plot(ts1, dist_scores1[:, 0], '--', label='assignment score\nchose by combined')
    # axes[1, 2].plot(ts1, dist_scores1[:, 1], '--', label='assignment score\nchose by combined')
    # axes[1, 2].axvline(1.4579229354858398, linestyle='-.', color='r', alpha=0.7, label='assignment change')
    # axes[1, 2].set_ylabel('dist-score')
    # axes[1, 2].legend()
    # axes[1, 3].plot(ts1, bw_scores_min1[:, 0], label='combined-min')
    # axes[1, 3].plot(ts1, bw_scores_min1[:, 1], label='combined-min')
    # axes[1, 3].plot(ts1, bw_scores1[:, 0], '--', label='assignment score\nchose by combined')
    # axes[1, 3].plot(ts1, bw_scores1[:, 1], '--', label='assignment score\nchose by combined')
    
    # axes[1, 3].axvline(1.4579229354858398, linestyle='-.', color='r', alpha=0.7, label='assignment change')
    # axes[1, 3].set_ylabel('bw-score')
    # axes[1, 3].legend()
    # axes[1, 4].plot(ts1, intf_scores_min1[:, 0], label='combined-min')
    # axes[1, 4].plot(ts1, intf_scores_min1[:, 1], label='combined-min')
    # axes[1, 4].plot(ts1, intf_scores1[:, 0], '--', label='assignment score\nchose by combined')
    # axes[1, 4].plot(ts1, intf_scores1[:, 1], '--', label='assignment score\nchose by combined')
    # axes[1, 4].axvline(1.4579229354858398, linestyle='-.', color='r', alpha=0.7, label='assignment change')
    # axes[1, 4].set_ylabel('intf-score')
    # axes[1, 4].legend()

    # plt.tight_layout()
    # plt.savefig('analysis-results/compare-score.pdf')

    
def generate_keys(locs, bws, helpees, schedulers=None):
    keys = []
    for l in locs:
        for b in bws:
            for h in helpees:
                if schedulers is not None:
                    for s in schedulers:
                        keys.append((l,b,s,h))
                else:
                    keys.append((l,b,h))
    return keys


def check_keys_matched(key, file_key):
    matched = True
    for k_element in key:
        if k_element not in file_key:
            matched = False
    return matched


def construct_result_based_on_keys(keys):
    result = {}
    for k,v in result_each_run.items():
        for key in keys:
            matched = check_keys_matched(key, k)
            if matched:
                if key in result.keys():
                    result[key] = np.hstack((result[key], v['all']))
                else:
                    result[key] = v['all']
    return result


def construct_full_frame_result_based_on_keys(keys):
    result = {}
    for k,v in result_each_run.items():
        for key in keys:
            matched = check_keys_matched(key, k)
            if matched:
                if key in result.keys():
                    result[key].append(v['full_frames'])
                else:
                    result[key] = [v['full_frames']]
    return result


def construct_frame_result_based_on_keys(keys):
    result = {}
    for k,v in result_each_run.items():
        for key in keys:
            matched = check_keys_matched(key, k)
            if matched:
                if key in result.keys():
                    result[key].append(v)
                else:
                    result[key] = [v]
    return result


def find_data_with_partial_keys(partial_keys, data):
    result = {}
    for k,v in data.items():
        matched = check_keys_matched(partial_keys, k)
        if matched:
            result[k] = v
    return result


def plot_dict_data_box(dict, name, idx):
    plt.figure()
    labels, data = dict.keys(), dict.values()
    ticks = []
    for label in labels:
        ticks.append(label[idx])
    x_positions = np.arange(len(result_each_run.keys()))
    cnt = 0
    for k in labels:
        plt.boxplot(dict[k], positions=np.array([x_positions[cnt]]), whis=(5, 95), autorange=True, showfliers=False)
        cnt += 1
    plt.xticks(range(0, len(ticks)), ticks, fontsize=15)
    plt.ylabel('Latency (s)')
    plt.tight_layout()
    plt.savefig('analysis-results/%s.png'%name)
  
   
def plot_full_frame(partial_results, name, idx):
    if partial_results != {}:
        fig = plt.figure()   
        labels, data = partial_results.keys(), partial_results.values() 
        print(labels)
        ticks = []
        for label in labels:
            ticks.append(label[idx])
        ax = fig.add_subplot(111)
        selected_threshold = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
        setting_to_diff_latency_frames, setting_to_diff_latency_frames_std = {}, {}
        setting_to_detection_rst, setting_to_detection_rst_std, setting_to_good_detection_frames \
            = {}, {}, {}
        setting_to_accuracy, setting_to_latency = {} ,{}
        cnt = 0
        for label in labels:
            # print(label)
            assert label[idx] == ticks[cnt]      
            setting_to_diff_latency_frames[label] = []
            setting_to_diff_latency_frames_std[label] = []
            setting_to_detection_rst[label] = []
            setting_to_detection_rst_std[label] = []
            setting_to_good_detection_frames[label] = []
            setting_to_accuracy[label] = []
            setting_to_latency[label] = []
            for t in selected_threshold:
                # calculate mean and std
                one_setting_num_full_frames = []
                # one_setting_mean_detect_area = []
                for one_run in partial_results[label]:
                    one_setting_num_full_frames.append(
                        get_percentage_frames_within_threshold(one_run, t, key='e2e-latency', num_nodes=num_nodes)
                    )
                    # setting_to_detection_rst[label].extend(one_run['detected_areas'])
                    # one_setting_mean_detect_area.append(np.mean(one_run['detected_areas']))            
                
                setting_to_diff_latency_frames[label].append(np.mean(one_setting_num_full_frames))
                setting_to_diff_latency_frames_std[label].append(np.std(one_setting_num_full_frames))

            one_setting_mean_detect_area, one_setting_acc, one_setting_latency = [], [], []
            for one_run in partial_results[label]:
                setting_to_detection_rst[label].extend(one_run['detected_areas'])
                one_setting_mean_detect_area.append(np.mean(one_run['detected_areas']))
                one_setting_acc.append(np.mean(one_run['detection_acc']))
                one_setting_latency.append(np.mean(one_run['e2e-latency']))
                # setting_to_good_detection_frames[label].append(
                #     get_num_frames_above_detected_space_threshold(one_run['detected_areas'], 5000)
                # )
            setting_to_accuracy[label] = one_setting_acc
            setting_to_latency[label] = one_setting_latency
            setting_to_detection_rst_std[label].append(np.std(one_setting_mean_detect_area))

            ax.errorbar(np.arange(1,len(selected_threshold)+1), setting_to_diff_latency_frames[label], \
                    yerr=setting_to_diff_latency_frames_std[label], capsize=2, color=sched_to_color[label[idx]], \
                    ls=linestyles[label[idx]], label=label[idx])
            print(label[idx], "Mean threshold value")
            print(setting_to_diff_latency_frames[label])
            print(setting_to_diff_latency_frames_std[label])
            cnt += 1
        
        ax.set_xticks(np.arange(1,len(selected_threshold)+1))
        ax.set_xticklabels(selected_threshold)

        plt.legend()
        plt.ylabel("% of Frames within latency")
        plt.xlabel('Latency (s)')
        plt.gca().set_ylim(bottom=0)
        plt.tight_layout()
        plt.savefig('analysis-results/%s-diff-latency.png'%name)

        fig = plt.figure(figsize=(9,5))
        ax = fig.add_subplot(111)

        # print(partial_results)
        plt_schemes = []
        for label in labels:
            plt_schemes.append('Harbor' if label[idx] == 'combined-adapt' else label[idx])
            label_str = "%s, %s"%('Harbor' if label[idx] == 'combined-adapt' else label[idx], '200 ms')
            ax.scatter(setting_to_diff_latency_frames[label][4], np.mean(setting_to_accuracy[label]),
            color=sched_to_color[label[idx]], label=label_str,
            marker=sched_to_marker[label[idx]])
            label_str = "%s, %s"%('Harbor' if label[idx] == 'combined-adapt' else label[idx], '500 ms')
            ax.scatter(setting_to_diff_latency_frames[label][7], np.mean(setting_to_accuracy[label]),
            edgecolor=sched_to_color[label[idx]], facecolors='none', marker=sched_to_marker[label[idx]],
            label=label_str)
            ax.plot([setting_to_diff_latency_frames[label][4], setting_to_diff_latency_frames[label][7]],
                [np.mean(setting_to_accuracy[label]), np.mean(setting_to_accuracy[label])], '--', \
                color=sched_to_color[label[idx]], linewidth=1)
            # ax.errorbar(np.mean(setting_to_latency[label]), np.mean(setting_to_accuracy[label]), \
            #     xerr=np.std(setting_to_latency[label]), yerr=np.std(setting_to_accuracy[label]), capsize=2,
            #     color=sched_to_color[label[idx]], marker=sched_to_marker[label[idx]])

        tablelegend(ax, ncol=len(plt_schemes), row_labels=['200 ms', '500 ms'],
                    col_labels=plt_schemes, title_label='threshold')
        
        plt.xlabel("# of frames within latency")
        plt.ylabel("Detection Accuracy")
        plt.tight_layout()
        plt.savefig('analysis-results/%s-two-dim.png'%name)

    
            

def plot_dict_data_cdf(dict, name, idx):
    plt.figure()
    labels, data = list(dict.keys()), list(dict.values())
    ticks = []
    for label_idx in range(len(labels)):
        ticks.append(labels[label_idx][idx])
        sns.ecdfplot(data[label_idx], label=labels[label_idx][idx])
    plt.legend()
    plt.xlabel('Latency (s)')
    plt.ylabel('CDF')
    plt.savefig('analysis-results/%s-cdf.png'%name)


def plot_based_on_setting(num_nodes):
    all_keys = generate_keys(LOC, BW, HELPEE, SCHEDULERS)

    result = construct_result_based_on_keys(all_keys)
    result_full_frame = construct_full_frame_result_based_on_keys(all_keys)
    result_all_frame = construct_frame_result_based_on_keys(all_keys)
    # print(result)
    combined_latency_improvement = {}
    for loc in LOC:
        for bw in BW:
            for helpee in HELPEE:
                partial_results = find_data_with_partial_keys((loc, bw, helpee), result)
                plot_dict_data_box(partial_results, str([loc, bw, helpee]), 2)
                plot_dict_data_cdf(partial_results, str([loc, bw, helpee]), 2)
                partial_results = find_data_with_partial_keys((loc, bw, helpee), result_all_frame)
                plot_full_frame(partial_results, str([loc, bw, helpee]), 2)
        
    # fig = plt.figure(figsize=(18, 9))
    # ax = fig.add_subplot(111)
    # cnt = 0
    # setting = []
    # for loc in LOC:
    #     for bw in BW:
    #         for helpee in HELPEE:
    #             setting.append((loc, bw, helpee))
    #             partial_results = find_data_with_partial_keys((loc, bw, helpee), result)

                
    #             schedulers = []
    #             combined_in_schedule_frames, other_sched_in_sched_frames = {}, {}
    #             for label in partial_results.keys():
    #                 schedulers.append(label[2])
    #             for label in partial_results.keys():
    #                 combined_in_schedule_frames[label[2]] =\
    #                     len(partial_results[label][partial_results[label] <= LATENCY_THRESHOLD])
    #                 if label[2] != 'combined':
    #                     other_sched_in_sched_frames[label[2]] = \
    #                         len(partial_results[label][partial_results[label] <= LATENCY_THRESHOLD])
                    
    #             latency_improvement = float(combined_in_schedule_frames['combined']-max(other_sched_in_sched_frames.values()))/ \
    #                                     max(other_sched_in_sched_frames.values())
    #             ax.bar(cnt, latency_improvement*100, align='center', alpha=0.5)
    #             cnt += 1
    # ax.set_xticks(np.arange(cnt))
    # # ax.set_xticklabels(setting)
    # map_setting = np.concatenate((np.arange(0, len(setting)).reshape(-1,1), np.array(setting).reshape(-1,3)), axis=1)
    # print(map_setting)
    # plt.ylabel('# of frames improvement over the best sched (%)')
    # np.savetxt("analysis-results/improvement_mapping.txt", map_setting, fmt='%s')
    # plt.savefig('analysis-results/combined-improvement.png')
    selected_schedulers = SCHEDULERS
    # selected_schedulers = ['combined', 'distributed']
    # for loc in LOC:
    #     for bw in BW:
    #         for helpee in HELPEE: 
    #             titles = []                
    #             print("lat ency-all-sched_figure:", (loc, bw, helpee))
    #             for i in range(num_nodes):
    #                 fig = plt.figure(figsize=(18,16)) 
                    
    #                 cnt = 1
                      
    #                 for sched in selected_schedulers:
    #                     ax = fig.add_subplot(len(selected_schedulers), 1, cnt)     
    #                     # folder = get_folder_based_on_setting('./', (loc, bw, helpee, sched))[0]
    #                     for k, v in result_each_run.items():
    #                         matched = check_keys_matched((loc, bw, helpee, sched), k)
    #                         if matched:
    #                             titles.append(k[0])
    #                             data_one_setting = v
    #                             break                     
    #                         ts, latency = construct_ts_latency_array(data_one_setting[i])
    #                         ax.plot(ts, latency, '--.', label="node%i"%i)
    #                         ax.legend()
    #                         ax.set_ylabel(sched, fontsize=20)

    #                 cnt += 1
    #                 plt.title(str(titles))
    #                 plt.xlabel('Time (s)')
    #                 plt.tight_layout()
    #                 print(titles)

    #                 plt.savefig('analysis-results/%s-node%d-latency.png'%(str((loc, bw, helpee)),cnt))



def plot_based_on_setting_multi():
    for loc in LOC:
        for bw in BW:
            for helpee in HELPEE:
                for sched in SCHEDULERS:
                    partial_results = find_data_with_partial_keys((loc, bw, helpee, sched),result_each_run)
                    plot_dict_data_box(partial_results, str([loc, bw, helpee, sched]), 3)
                    plot_dict_data_cdf(partial_results, str([loc, bw, helpee, sched]), 3)
                    



def get_per_experiment_stats(result_dir, node_num):
    stats = {}
    for i in range(node_num):
        stats[i] = np.loadtxt(result_dir+'/node%d_delay.txt'%i)
    stats['all'] = np.loadtxt(result_dir+'/all_delay.txt')
    return stats


def get_folder_based_on_setting(data_dir, setting):
    if not os.path.exists('analysis-results/setting_to_folder.json'):
        get_all_runs_results(data_dir, 'data-')
    # if os.path.exists('analysis-results/setting_to_folder.json'):
    data = open('analysis-results/setting_to_folder.json', 'rb').read()
    data = json.loads(data)
    
    result_dir = []
    for key, dir_value in data.items():
        key_tuple = eval(key)
        # print(key)
        matched = check_keys_matched(setting, key_tuple)
        if matched:
            result_dir.append(dir_value)
    
    return result_dir
                

def get_all_runs_results(data_dir, key, with_ssim=False, parse_exp_stats=True):
    global num_nodes
    # TODO: cuurently node number has to start on 0, support node number to be largely different (e.g. 0, 145, etc)
    dirs = os.listdir(data_dir)
    for dir in dirs:
        if key in dir:
            # put this is a function def 
            config = run_experiment.parse_config_from_file(data_dir+dir+'/config.txt')
            conf_key = get_key_from_config(config, dir)
            # insert (sched, network, mobility, helpee) in to config_set
            config_set.add((num_nodes, config["network_trace"], config["location_file"], \
                config["helpee_conf"], config["t"]))
            
            if parse_exp_stats:
                result_each_run[conf_key] = get_stats_on_one_run(data_dir+dir, num_nodes,\
                    config["helpee_conf"], config, with_ssim=with_ssim)[0]
                # print(conf_key, result_each_run[conf_key]['sent_frames'])
            setting_to_folder[str(conf_key)] = dir
    with open('analysis-results/setting_to_folder.json', 'w') as f:
        json.dump(setting_to_folder, f)

def plot_bar_across_runs():
    fig = plt.figure(figsize=(50, 8))
    ax = fig.add_subplot(111)
    x_positions = np.arange(len(result_each_run.keys()))
    cnt = 0
    for k, v in result_each_run.items():
        ax.boxplot(v['all'], positions=np.array([x_positions[cnt]]), whis=(5, 95), autorange=True, showfliers=False)
        cnt += 1
    plt.ylabel('Latency (s)')
    plt.savefig('analysis-results/all_runs.png')


def plot_bar_based_on_schedule(schedule):
    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(111)
    x_positions = np.arange(len(result_each_run.keys()))
    ticks = []
    cnt = 0
    for k, v in result_each_run.items():
        if schedule in k:
            ticks.append(k)
            ax.boxplot(v['all'], positions=np.array([x_positions[cnt]]), whis=(5, 95), autorange=True, showfliers=False)
        cnt += 1
    # ax.set_xticklabels(ticks)
    plt.ylabel('Latency (s)')
    plt.savefig('analysis-results/%s.png'%schedule)


def plot_bars_compare_schedules(schedules):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    x_positions = np.arange(len(schedules))
    cnt = 0
    schedule_data, schedule_overhead, server_compute_time, schedule_dl_latency = {}, {}, {}, {}
    schedule_helpee_data, schedule_helper_data = {}, {}
    schedule_to_frames_within_threshold = {}
    sched_to_latency = {}
    schedule_to_detected_spaces, sched_to_acc = {}, {}
    for schedule in schedules:
        for k, v in result_each_run.items():
            if schedule in k:
                if schedule not in schedule_data.keys():
                    schedule_overhead[schedule] = v['overhead']
                    # schedule_dl_latency[schedule] = v['dl-latency']
                    server_compute_time[schedule] = v['sched_latency']
                    schedule_data[schedule] = v['all']
                    schedule_helpee_data[schedule] = v['helpee']
                    schedule_helper_data[schedule] = v['helper']
                    schedule_to_frames_within_threshold[schedule] = [get_percentage_frames_within_threshold(v, LATENCY_THRESHOLD, SSIM_THRESHOLD)]
                    schedule_to_detected_spaces[schedule] = [np.mean(v['detected_areas'])]
                    sched_to_acc[schedule] = [np.mean(v['detection_acc'])]
                    sched_to_latency[schedule] = [np.mean(v['e2e-latency'])]
                else:
                    schedule_overhead[schedule] = np.hstack((schedule_overhead[schedule], v['overhead']))
                    # schedule_dl_latency[schedule] = np.hstack((schedule_dl_latency[schedule], v['dl-latency']))
                    schedule_data[schedule] = np.hstack((schedule_data[schedule], v['all']))
                    schedule_helpee_data[schedule] = np.hstack((schedule_helpee_data[schedule], v['helpee']))
                    schedule_helper_data[schedule] = np.hstack((schedule_helper_data[schedule], v['helper']))
                    server_compute_time[schedule] = np.hstack((server_compute_time[schedule], v['sched_latency']))
                    schedule_to_frames_within_threshold[schedule] = \
                        np.hstack((schedule_to_frames_within_threshold[schedule], \
                            get_percentage_frames_within_threshold(v, LATENCY_THRESHOLD, SSIM_THRESHOLD)))
                    schedule_to_detected_spaces[schedule].append(np.mean(v['detected_areas']))
                    sched_to_acc[schedule].append(np.mean(v['detection_acc']))
                    sched_to_latency[schedule].append(np.mean(v['e2e-latency']))
    for schedule in schedule_data.keys():
        ax.boxplot(schedule_data[schedule], positions=np.array([x_positions[cnt]-0.2]), whis=(5, 95), autorange=True, showfliers=False)
        ax.boxplot(schedule_helpee_data[schedule], positions=np.array([x_positions[cnt]]), whis=(5, 95), autorange=True, showfliers=False)
        ax.boxplot(schedule_helper_data[schedule], positions=np.array([x_positions[cnt]+0.2]), whis=(5, 95), autorange=True, showfliers=False)
        cnt += 1
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(schedules)
    plt.ylabel('Latency (s)')
    plt.savefig('analysis-results/schedule_compare.png')

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    cnt = 0
    for schedule in schedule_data.keys():
        ax.boxplot(schedule_overhead[schedule], positions=np.array([x_positions[cnt]]), whis=(5, 95), autorange=True, showfliers=False)
        cnt += 1
    ax.set_xticks(x_positions)
    ax.set_xticklabels(schedules)
    plt.ylabel('Computational Overhead (s)')
    plt.savefig('analysis-results/schedule_overhead.png') 
    
    # fig = plt.figure(figsize=(9,6))
    # ax = fig.add_subplot(111)
    # cnt = 0
    # for schedule in schedule_data.keys():
    #     ax.boxplot(schedule_dl_latency[schedule], positions=np.array([x_positions[cnt]]), whis=(5, 95), autorange=True, showfliers=False)
    #     cnt += 1
    # ax.set_xticks(x_positions)
    # ax.set_xticklabels(schedules)
    # plt.ylabel('DL latency (s)')
    # plt.savefig('analysis-results/schedule_dl_latency.png')     


    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    cnt = 0
    for schedule in schedule_data.keys():
        ax.boxplot(server_compute_time[schedule], positions=np.array([x_positions[cnt]]), whis=(5, 95), autorange=True, showfliers=False)
        cnt += 1
    ax.set_xticks(x_positions)
    ax.set_xticklabels(schedules)
    plt.ylabel('Time to compute schedule (s)')
    plt.savefig('analysis-results/server_schedule_time.png')      
    

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    cnt = 0
    for schedule in schedule_data.keys():
        # print(schedule)
        ax.boxplot(schedule_to_frames_within_threshold[schedule], positions=np.array([x_positions[cnt]]), whis=(5, 95), autorange=True, showfliers=False)
        cnt += 1
    ax.set_xticks(x_positions)
    ax.set_xticklabels(schedules)
    if SSIM_THRESHOLD is None:
        plt.ylabel('Percentage of frame in schedule (%3fs)'%LATENCY_THRESHOLD)
    else:
        plt.ylabel('Percentage of frame in schedule (%3f s, %3f SSIM)'%(LATENCY_THRESHOLD, SSIM_THRESHOLD))
    plt.tight_layout()
    plt.savefig('analysis-results/schedule_frames_within_latency.png')

    # fig = plt.figure(figsize=(24,8))
    # cnt = 1
    # for schedule in schedule_data.keys():
    #     ax = fig.add_subplot(1, len(schedules), cnt)
    #     # bins = np.arange(0, 1, 0.2)
    #     arr = ax.hist(schedule_data[schedule], bins=4, cumulative=True)
    #     ax.set_xlabel(schedule)
    #     # for i in range(len(bins)-1):
    #     #     plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
    #     cnt += 1
    # plt.xticks(np.arange(0,4,1))
    # plt.gca().set(title='Frame latency frequency Histogram', ylabel='# of frames')
    # plt.xlabel('Latency (s)')
    # plt.savefig('analysis-results/frames_latency_histogram.png')
    
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    sched_to_different_latency_cnts = {}
    selected_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
    for schedule in schedules:
        sched_to_different_latency_cnts[schedule] = {}
        for t in selected_thresholds:
            sched_to_different_latency_cnts[schedule][t] = []
        for k, v in result_each_run.items():
            if schedule in k:
                for t in selected_thresholds:
                    sched_to_different_latency_cnts[schedule][t].append(
                        get_percentage_frames_within_threshold(v, t, key='e2e-latency', num_nodes=num_nodes)
                    )
    sched_to_different_latency_mean = {}
    sched_to_different_latency_std = {}
    for schedule in schedules:
        sched_to_different_latency_mean[schedule] = []
        sched_to_different_latency_std[schedule] = []
        for t in selected_thresholds:
            sched_to_different_latency_mean[schedule].append(np.mean(sched_to_different_latency_cnts[schedule][t]))
            sched_to_different_latency_std[schedule].append(np.std(sched_to_different_latency_cnts[schedule][t]))
        # print("*******")
        # print(schedule, sched_to_different_latency_mean[schedule])
        # print(sched_to_different_latency_std[schedule][1])
        ax.plot(np.arange(0, len(selected_thresholds)), sched_to_different_latency_mean[schedule], '-o', label=schedule if schedule is not 'combined-loc' else 'combined-loc-3.5m')
        ax.set_xticks(np.arange(0, len(selected_thresholds)))
        ax.set_xticklabels(selected_thresholds)
    plt.legend()
    plt.ylabel("% of Frames within the latency threshold")
    plt.gca().set_ylim(bottom=0)
    plt.xlabel('Latency (s)')
    plt.savefig('analysis-results/frames_latency_diff_threshold.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for schedule in schedule_data.keys():
        ax.scatter(sched_to_different_latency_mean[schedule][1], np.mean(schedule_to_detected_spaces[schedule]),
        label='Harbor' if schedule == 'combined-adapt' else schedule, marker=sched_to_marker[schedule],
        color=sched_to_color[schedule])
        ax.scatter(sched_to_different_latency_mean[schedule][4], np.mean(schedule_to_detected_spaces[schedule]),
        label='Harbor' if schedule == 'combined-adapt' else schedule, marker=sched_to_marker[schedule],
        color=sched_to_color[schedule])
        # ax.errorbar(sched_to_different_latency_mean[schedule][1], np.mean(schedule_to_detected_spaces[schedule]), \
        #     xerr=sched_to_different_latency_std[schedule][1], yerr=np.std(schedule_to_detected_spaces[schedule]), capsize=2,
        #     color=sched_to_color[schedule], marker=sched_to_marker[schedule])
        # print(schedule, sched_to_different_latency_mean[schedule][1], np.mean(schedule_to_detected_spaces[schedule]))
        # print(schedule, sched_to_different_latency_std[schedule][1], np.std(schedule_to_detected_spaces[schedule]))

    plt.ylim([2000, 5200])
    plt.xlabel("# of frame within latency threshold")
    plt.ylabel("Detected space ($m^2$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis-results/aggregated-two-dim.png')

    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    ims, plot_schedules = [], []
    for schedule in schedule_data.keys():
        plot_schedules.append('Harbor' if schedule == 'combined-adapt' else schedule)
        im = ax.scatter(sched_to_different_latency_mean[schedule][1], np.mean(sched_to_acc[schedule]),
            marker=sched_to_marker[schedule], color=sched_to_color[schedule])
        ims.append(im)
        im = ax.scatter(sched_to_different_latency_mean[schedule][4], np.mean(sched_to_acc[schedule]),
            marker=sched_to_marker[schedule], edgecolor=sched_to_color[schedule], facecolors='none')
        ims.append(im)
        ax.plot([sched_to_different_latency_mean[schedule][1], sched_to_different_latency_mean[schedule][4]],
            [np.mean(sched_to_acc[schedule]), np.mean(sched_to_acc[schedule])], '--', color=sched_to_color[schedule], linewidth=1)
        # ax.errorbar(sched_to_different_latency_mean[schedule][1], np.mean(sched_to_acc[schedule]), \
        #     xerr=sched_to_different_latency_std[schedule][1], yerr=np.std(sched_to_acc[schedule]), capsize=2,
        #     color=sched_to_color[schedule])
        # print(schedule, sched_to_different_latency_mean[schedule][1], np.mean(sched_to_acc[schedule]))
        # print(schedule, sched_to_different_latency_std[schedule][1], np.std(sched_to_acc[schedule]))


    # create blank rectangle
    extra = Rectangle((0, 0), 0.5, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

    #Create organized list containing all handles for table. Extra represent empty space
    legend_handle = [extra, extra, extra]
    for index in range(int(len(ims)/2)):
        legend_handle += [extra, ims[2*index], ims[2*index+1]]

    #Define the labels
    label_row_1 = [r"threshold", r"200 ms", r"500 ms"]
    label_arr = label_row_1
    for scheme in plot_schedules:
        label_arr += [scheme, "", ""]
    
    #organize labels for table construction
    legend_labels = np.array(label_arr)
    # legend_labels = np.concatenate([label_row_1, label_j_1, label_empty * 2, label_j_2, label_empty * 2, label_j_3, label_empty * 2])

    #Create legend
    ax.legend(legend_handle, legend_labels,  ncol = len(plot_schedules) + 1, shadow = True, handletextpad = -2, fontsize=14)

    plt.ylim([0.5, 0.75])
    plt.xlabel("# of frame within latency threshold")
    plt.ylabel("Detection Accuracy")
    # plt.legend()
    plt.tight_layout()
    plt.savefig('analysis-results/aggregated-two-dim-acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for schedule in schedule_data.keys():
        ax.scatter(np.mean(sched_to_latency[schedule]), np.mean(sched_to_acc[schedule]),
            label='Harbor' if schedule == 'combined-adapt' else schedule, marker=sched_to_marker[schedule],\
            color=sched_to_color[schedule])
        ax.errorbar(np.mean(sched_to_latency[schedule]), np.mean(sched_to_acc[schedule]), \
            xerr=np.std(sched_to_latency[schedule]), yerr=np.std(sched_to_acc[schedule]), capsize=2,
            color=sched_to_color[schedule])
        # print(schedule, np.mean(sched_to_latency[schedule]), np.mean(sched_to_acc[schedule]))

    plt.ylim([0.5, 0.75])
    plt.xlabel("Frame Latency (s)")
    plt.ylabel("Detection Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis-results/aggregated-two-dim-acc-latency.png')

    sched_to_different_metric_cnts = {}
    latency_thresholds = [0.1, 0.2, 0.3]
    space_thresholds = [3000, 4000, 4500, 5000]
    for schedule in schedules:
        sched_to_different_metric_cnts[schedule] = {}
        for t in latency_thresholds:
            for s in space_thresholds:
                sched_to_different_latency_cnts[schedule][(t,s)] = []
        for k, v in result_each_run.items():
            if schedule in k:
                for t in latency_thresholds:
                    for s in space_thresholds:
                        sched_to_different_latency_cnts[schedule][(t,s)].append(
                            get_num_frames_within_latency_above_detected_space(v, t, s)
                        )
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    cnt = 0
    xticks = []
    selected_metric = (0.2, 3000)
    for schedule in sched_to_different_latency_cnts:
        xticks.append('Harbor' if schedule == 'combined-adapt' else schedule)
        ax.bar(cnt, np.mean(sched_to_different_latency_cnts[schedule][selected_metric]),
            align='center', label='Harbor' if schedule == 'combined-adapt' else schedule, color=sched_to_color[schedule],
            alpha=0.5)
        ax.errorbar(cnt, np.mean(sched_to_different_latency_cnts[schedule][selected_metric]),
            yerr=np.std(sched_to_different_latency_cnts[schedule][selected_metric]), capsize=3,
            color=sched_to_color[schedule])
        cnt += 1
    plt.xticks(np.arange(len(sched_to_different_latency_cnts.keys())), xticks)
    plt.ylabel("# of frames with\n(%0.2f s latency, %d $m^2$ detected space)" % selected_metric)
    plt.tight_layout()
    plt.savefig('analysis-results/frames-%0.2flatency-%dspace.png'%selected_metric)


    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    cnt = 0
    xticks = []
    for schedule in sorted(schedule_data.keys()):
        xticks.append('pure-V2I' if schedule == 'emp' else schedule)
        ax.bar(cnt, np.mean(schedule_to_detected_spaces[schedule]), align='center', 
            label='pure-V2I' if schedule == 'emp' else schedule, color=sched_to_color[schedule],
            alpha=0.5)
        cnt += 1
    plt.xticks(np.arange(len(schedule_data.keys())), xticks)
    plt.ylim([4000, 5200])
    plt.ylabel("Detected space ($m^2$)")
    plt.tight_layout()
    plt.savefig('analysis-results/detected_space.png')

def plot_bar_compare_encode():
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    x_positions = np.arange(2)
    encode_data = {}
    ENCODE = ['0', '1']
    labels = ['No Adaptive Encoding', 'Adaptive Encoding']
    cnt = 0
    for encode_type in ENCODE:
        for k, v in result_each_run.items():
            if encode_type in k:
                if encode_type not in encode_data.keys():
                    encode_data[encode_type] = \
                        get_num_frames_within_threshold(v, LATENCY_THRESHOLD, SSIM_THRESHOLD)
                else:
                    encode_data[encode_type] = \
                        np.hstack((encode_data[encode_type], \
                            get_num_frames_within_threshold(v, LATENCY_THRESHOLD, SSIM_THRESHOLD)))
    for encode_type in encode_data.keys():
        ax.boxplot(encode_data[encode_type], positions=np.array([x_positions[cnt]]), whis=(5, 95),\
            autorange=True, showfliers=False)
        cnt += 1
    ax.set_xticklabels(labels)
    plt.ylabel('# of Frame in schedule (%3f s, %3f SSIM)'%(LATENCY_THRESHOLD, SSIM_THRESHOLD))
    plt.savefig('analysis-results/compare_encoding_types.png')
        

def calculate_per_node_mean(setting):
    print("Setting: %s" % str(setting))
    node_result = {}
    for k, v in result_each_run.items():
        matched = check_keys_matched(setting, k)
        if matched:
            print(k)
            for i in range(num_nodes):
                node_latency = np.array(sorted(v[i].values()))[:, 0]
                if i in node_result.keys():
                    node_result[i].append(np.mean(node_latency))
                else:
                    node_result[i] = [np.mean(node_latency)]
   
    return node_result


def calculate_per_node_per_frame_mean(setting):
    frame_result = {}
    for k, v in result_each_run.items():
        matched = check_keys_matched(setting, k)
        if matched:
            for i in range(num_nodes):
                node_latency = v[i]
                if i in frame_result.keys():
                    frame_result[i] = np.vstack((frame_result[i], node_latency))
                else:
                    frame_result[i] = node_latency
    frame_mean = {}
    frame_std = {}
    for node, v in frame_result.items():
        frame_mean[node] = np.mean(v, axis=0)
        frame_std[node] = np.std(v, axis=0)
    return frame_mean, frame_std


def plot_bar(data, name):
    if len(data.values()) == 0:
        return None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(data.values()))
    values = data.values()
    cnt = 0
    print(name)
    for value in values:
        mean, std = np.mean(value), np.std(value)
        print(std)
        x = np.arange(cnt, (cnt+1))
        plt.errorbar(x, mean, yerr=std, capsize=4)
        ax.scatter(x, mean)
        cnt += 1
    
    plt.ylabel('Average Latency (s)')
    plt.savefig('analysis-results/%s-per-node.png'%name)


def plot_per_frame(mean, std, name):
    total_num = len(mean.keys())
    cnt = 1
    fig = plt.figure(figsize=(30, 18))
    for k in mean.keys():
        ax = fig.add_subplot(total_num, 1, cnt)
        m, s = mean[k], std[k]
        frame_idx = np.arange(len(m))
        ax.errorbar(frame_idx, m, fmt = 'o', yerr=s, capsize=2, label='node%d'%k)
        ax.legend()
        cnt += 1
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Average Latency (s)')
    plt.xlabel('Frame Number')
    # plt.legend()
    plt.tight_layout()
    plt.savefig('analysis-results/%s-frame-node.png'%(name))

def repeat_exp_analysis():
    all_keys = generate_keys(LOC, BW, HELPEE, SCHEDULERS)
    for key in all_keys:
        node_result = calculate_per_node_mean(key)
        frame_mean, frame_std = calculate_per_node_per_frame_mean(key)
        plot_per_frame(frame_mean, frame_std, str(key))
        # plot_bar(node_result, str(key))

        
def analyze_msg_overhead():
    # msg size overhead
    sched_to_num_nodes_to_msg_overhead = {}
    sched_to_num_nodes_to_performance_overhead = {}
    for key in result_each_run.keys():
        data_dir = key[0]
        config = run_experiment.parse_config_from_file(data_dir + '/config.txt')
        num_nodes, sched = int(config['num_of_nodes']), config['scheduler']
        if 'fixed' in sched:
            sched = 'fixed'
        if sched not in sched_to_num_nodes_to_msg_overhead.keys():
            sched_to_num_nodes_to_msg_overhead[sched] = {}
            sched_to_num_nodes_to_performance_overhead[sched] = {}
        if num_nodes not in sched_to_num_nodes_to_msg_overhead[sched].keys():
            sched_to_num_nodes_to_msg_overhead[sched][num_nodes] = []
            sched_to_num_nodes_to_performance_overhead[sched][num_nodes] = []
        sched_to_num_nodes_to_msg_overhead[sched][num_nodes].append(get_control_msg_data_overhead(data_dir, num_nodes))
        sched_to_num_nodes_to_performance_overhead[sched][num_nodes].append(np.mean(result_each_run[key]['all']))
    
    fig, axes = plt.subplots(2, 1, sharex=True)
    # get baseline performance
    nodes_base, perf_mean_base, perf_std_base = [], [], []
    for node, performance_overhead in sorted(sched_to_num_nodes_to_performance_overhead['fixed'].items()):
        # print(node, performance_overhead)
        nodes_base.append(node)
        perf_mean_base.append(np.mean(performance_overhead))
        perf_std_base.append(np.std(performance_overhead))
    for sched, nodes_to_msg_overhead in sched_to_num_nodes_to_msg_overhead.items():
        nodes, overhead_mean, overhead_std = [], [], []
        for node, overheads in sorted(nodes_to_msg_overhead.items()):
            nodes.append(node)
            overhead_mean.append(np.mean(overheads))
            overhead_std.append(np.std(overheads))
        axes[0].errorbar(nodes, overhead_mean, yerr=overhead_std, capsize=2, label=sched)
        # print('data overhead', overhead_mean, overhead_std)
        nodes, overhead_mean, overhead_std = [], [], []
        for node, performance_overhead in sorted(sched_to_num_nodes_to_performance_overhead[sched].items()):
            nodes.append(node)
            overhead_mean.append(np.mean(performance_overhead))
            overhead_std.append(np.std(performance_overhead))
        # print("perf_overhead", overhead_mean, perf_mean_base, overhead_std, perf_std_base)
        perf_improvement_mean, perf_improvement_std = 100. * (np.array(overhead_mean)/np.array(perf_mean_base) - 1), \
            np.array(overhead_std)/np.array(perf_mean_base)
        # print("percentage std ", perf_improvement_std)

        axes[1].errorbar(nodes, perf_improvement_mean, yerr=perf_improvement_std, capsize=2, label=sched)
    axes[0].set_ylabel('Data overhead (%)')
    axes[0].legend()
    axes[1].set_ylabel('Avg latency\nincrease (%)')
    axes[1].set_xlabel('Number of nodes (helpees)')
    axes[1].set_xticks([2,3,4,5,6])
    axes[1].set_xticklabels(['2(1)', '3(1)', '4(2)', '5(2)', '6(2)'])
    fig.tight_layout()
    plt.savefig('analysis-results/overhead.png')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default="~/v2x/", type=str, help="data directory")
    parser.add_argument('-p', '--prefix', default='data-', type=str, help='prefix on data dir to analyze')
    ## high level task to anal e.g. compare_scheduler, compare_effect_multi, 
    ## one-help-many can be parsed from config
    parser.add_argument('-t', '--time_threshold', default=0.2, type=float, help='threshold to evaluate a good frame latency')
    parser.add_argument('--task', default=[], nargs='+', type=str, help='additional analysis task to do (ssim|compare_adaptive_encode|get_folder_on_setting)')
    parser.add_argument('--ssim_threshold', default=None, type=float, help='threshold to evaluate a good SSIM')

    args = parser.parse_args()

    data_dir = args.data_dir
    key = args.prefix
    task = args.task
    global LATENCY_THRESHOLD, SSIM_THRESHOLD
    with_ssim = False
    if task == 'ssim':
        with_ssim = True
        SSIM_THRESHOLD = 0.6
    LATENCY_THRESHOLD = args.time_threshold
    SSIM_THRESHOLD = args.ssim_threshold

    # create a analysis-results/ dir under data_dir
    
    # rst = get_folder_based_on_setting(data_dir, ('helpee-last-two',))
    # print(rst)
    # for folder in rst:
    #     print("deleting %s"%folder)
    #     os.system("rm -rf %s"%folder)
    
    compare_sched_in_one_plot, compare_deadline = False, False
    parse_exp_stats = True 
    msg_overhead_analyze = False
    if len(args.task)>0:
        if args.task[0] == 'get_folder_on_setting':        
            print(tuple(args.task[1:]))
            rst = get_folder_based_on_setting(data_dir, tuple(args.task[1:]))
            print(rst)
            return
        elif args.task[0] == 'compare_sched':
            compare_sched_in_one_plot = True
            sched1_folder, sched2_folder = args.task[1], args.task[2]
        elif args.task[0] == 'plot_settings_summary':
            parse_exp_stats = False
        elif args.task[0] == 'analyze_msg_overhead':
            msg_overhead_analyze = True
        elif args.task[0] == 'compare_deadline':
            compare_deadline = True
            sched1_folder, sched2_folder = args.task[1], args.task[2]
        

    
    os.system('mkdir %s/analysis-results/'%data_dir)

    # # read all exp data, need a return value 
    get_all_runs_results(data_dir, key, with_ssim=with_ssim, parse_exp_stats=parse_exp_stats)

    if compare_deadline:
        pass
        return

    if not parse_exp_stats:
        get_summary_of_settings(config_set)
        return

    if compare_sched_in_one_plot:
        compare_two_sched(sched1_folder, sched2_folder)
        return 
    
    if msg_overhead_analyze:
        analyze_msg_overhead()
        return
    
    # compare schedule
    schedules = SCHEDULERS
    plot_bars_compare_schedules(schedules) # plot_bars_comapring_schedules(SCHEDULERS)

    # plot_compare_schedule_by_setting()
    # name of figure 'compare-schedule-by-[set]'
    plot_based_on_setting(num_nodes)
    
    
    # 111 lte-28 helpee-start-middle

    # plot_compare_latency_of_settings(set_of_settings)
    # repeat_exp_analysis()
    
    # rst = get_folder_based_on_setting(data_dir, ('111', 'lte-28', 'helpee-start-middle', 'combined'))
    # print(rst)



if __name__ == '__main__':
    main()
