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
from util import *

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


## TODO: define a function that plot comparison for only two schedulers
## with each node's latecy and assignment

def compare_two_sched(setting1, setting2):
    result_sched1, result_sched2 = result_each_run[setting1], result_each_run[setting2]
    

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
    fig = plt.figure()   
    labels, data = partial_results.keys(), partial_results.values() 
    print(labels)
    ticks = []
    for label in labels:
        ticks.append(label[idx])
    ax = fig.add_subplot(111)
    selected_threshold = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
    setting_to_diff_latency_frames, setting_to_diff_latency_frames_std = {}, {}
    cnt = 0
    for label in labels:
        assert label[idx] == ticks[cnt]
        
        setting_to_diff_latency_frames[label] = []
        setting_to_diff_latency_frames_std[label] = []
        for t in selected_threshold:
            # calculate mean and std
            one_setting_num_full_frames = []
            for one_run in partial_results[label]:
                one_setting_num_full_frames.append(get_percentage_frames_within_threshold(one_run, t))
            setting_to_diff_latency_frames[label].append(np.mean(one_setting_num_full_frames))
            setting_to_diff_latency_frames_std[label].append(np.std(one_setting_num_full_frames))
        ax.errorbar(np.arange(1,len(selected_threshold)+1), setting_to_diff_latency_frames[label], \
                yerr=setting_to_diff_latency_frames_std[label], capsize=2,
                label=label[idx])
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
    print(result)
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
                
    
        

def get_all_runs_results(data_dir, key, with_ssim=False):
    global num_nodes
    # TODO: cuurently node number has to start on 0, support node number to be largely different (e.g. 0, 145, etc)
    dirs = os.listdir(data_dir)
    for dir in dirs:
        if key in dir:
            # put this is a function def 
            config = run_experiment.parse_config_from_file(data_dir+dir+'/config.txt')
            scheduler = config["scheduler"]
            network = config["network_trace"].split('/')[-1][:-4] # TODO: do we really need to split?
            mobility = config["location_file"].split('/')[-1][:-4]
            helpee = config["helpee_conf"].split('/')[-1][:-4] # TODO: helpe_config instead of helpee
            num_nodes = int(config["num_of_nodes"])
            adapt_frame_skipping = int(config["adapt_frame_skipping"])
            if adapt_frame_skipping == 1:
                scheduler += '-adapt'
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

            # insert (sched, network, mobility, helpee) in to config_set
            config_set.add((num_nodes, config["network_trace"], config["location_file"], \
                config["helpee_conf"], config["t"]))
            
            result_each_run[conf_key] = get_stats_on_one_run(data_dir+dir, num_nodes,\
                config["helpee_conf"], with_ssim=with_ssim)[0]
            print(conf_key, result_each_run[conf_key]['sent_frames'])
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
    schedule_data = {}
    schedule_helpee_data, schedule_helper_data = {}, {}
    schedule_to_frames_within_threshold = {}
    for schedule in schedules:
        for k, v in result_each_run.items():
            if schedule in k:
                if schedule not in schedule_data.keys():
                    schedule_data[schedule] = v['all']
                    schedule_helpee_data[schedule] = v['helpee']
                    schedule_helper_data[schedule] = v['helper']
                    schedule_to_frames_within_threshold[schedule] = [get_percentage_frames_within_threshold(v, LATENCY_THRESHOLD, SSIM_THRESHOLD)]
                else:
                    schedule_data[schedule] = np.hstack((schedule_data[schedule], v['all']))
                    schedule_helpee_data[schedule] = np.hstack((schedule_helpee_data[schedule], v['helpee']))
                    schedule_helper_data[schedule] = np.hstack((schedule_helper_data[schedule], v['helper']))
                    schedule_to_frames_within_threshold[schedule] = \
                        np.hstack((schedule_to_frames_within_threshold[schedule], \
                            get_percentage_frames_within_threshold(v, LATENCY_THRESHOLD, SSIM_THRESHOLD)))
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
        print(schedule)
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
                        get_percentage_frames_within_threshold(v, t, SSIM_THRESHOLD)
                    )
    sched_to_different_latency_mean = {}
    sched_to_different_latency_std = {}
    for schedule in schedules:
        sched_to_different_latency_mean[schedule] = []
        sched_to_different_latency_std[schedule] = []
        for t in selected_thresholds:
            sched_to_different_latency_mean[schedule].append(np.mean(sched_to_different_latency_cnts[schedule][t]))
            sched_to_different_latency_std[schedule].append(np.std(sched_to_different_latency_cnts[schedule][t]))
        ax.plot(np.arange(0, len(selected_thresholds)), sched_to_different_latency_mean[schedule], label=schedule)
        # ax.errorbar(np.arange(0, len(selected_thresholds)), sched_to_different_latency_mean[schedule], yerr=sched_to_different_latency_std[schedule], capsize=2, label=schedule)
        ax.set_xticks(np.arange(0, len(selected_thresholds)))
        ax.set_xticklabels(selected_thresholds)
    plt.legend()
    plt.ylabel("% of Frames within the latency threshold")
    plt.gca().set_ylim(bottom=0)
    plt.xlabel('Latency (s)')
    plt.savefig('analysis-results/frames_latency_diff_threshold.png')
        

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
   
    # print(node_result)
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
        # frame_mean, frame_std = calculate_per_node_per_frame_mean(key)
        # plot_per_frame(frame_mean, frame_std, str(key))
        # print("node results")
        # print(node_result)
        plot_bar(node_result, str(key))

        


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--type', default="multi", type=str, help="analyze type multi|single")
    parser.add_argument('-d', '--data_dir', default="~/v2x/", type=str, help="data directory")
    # parser.add_argument('-n', '--num_nodes', default=6, type=int, help="number of nodes")
    # parser.add_argument('-f', '--frames', default=80, type=int, help='number of frames considered')
    parser.add_argument('-p', '--prefix', default='data-', type=str, help='prefix on data dir to analyze')
    ## high level task to anal e.g. compare_scheduler, compare_effect_multi, 
    ## one-help-many can be parsed from config
    # parser.add_argument('-m', '--multi', default=False, type=bool, help='compare multi helper')  # delete this arg
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
    
    # rst = get_folder_based_on_setting(data_dir, ('111', 'lte-28', 'helpee-start-middle'))
    # print(rst)
    # for folder in rst:
    #     os.system("rm -rf %s"%folder)
    
    if len(args.task)>0:
        if args.task[0] == 'get_folder_on_setting':        
            print(tuple(args.task[1:]))
            rst = get_folder_based_on_setting(data_dir, tuple(args.task[1:]))
            print(rst)
            return
    
    os.system('mkdir %s/analysis-results/'%data_dir)

    # # read all exp data, need a return value 
    get_all_runs_results(data_dir, key, with_ssim=with_ssim)
    get_summary_of_settings(config_set)

    # plot_bar_across_runs()

    # for sched in SCHEDULERS:
    #     # plot_bars_of_a_schedule(sched)
    #     plot_bar_based_on_schedule(sched)
    
    # compare schedule
    # schedules = ['combined', 'distributed']
    schedules = SCHEDULERS
    plot_bars_compare_schedules(schedules) # plot_bars_comapring_schedules(SCHEDULERS)

    # plot_compare_schedule_by_setting()
    # name of figure 'compare-schedule-by-[set]'
    plot_based_on_setting(num_nodes)
    
    
    # 111 lte-28 helpee-start-middle

    # plot_compare_latency_of_settings(set_of_settings)
    # repeat_exp_analysis()
    
    rst = get_folder_based_on_setting(data_dir, ('111', 'lte-28', 'helpee-start-middle', 'combined'))
    print(rst)



if __name__ == '__main__':
    main()
