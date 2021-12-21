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
from analyze import generate_keys, check_keys_matched

plt.rc('font', family='sans-serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})

setting_to_folder = {}
result_each_run = {}
SCHEDULERS = []
LOC = []
BW = []
HELPEE = []
config_set = set()
num_nodes = 6
all_bw = []
helpee_disconnection = []
all_dist = []
import scipy

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def get_setting_characteristic(setting):
    num_nodes, bw_file, loc, helpee_conf, run_time = int(setting[0]), setting[1], setting[2], setting[3], int(setting[4])
    if 'mn-wifi' in bw_file:
        bw_file = bw_file.replace('mn-wifi', 'mininet-wifi')
    if 'mn-wifi' in loc:
        loc = loc.replace('mn-wifi', 'mininet-wifi')
    if 'mn-wifi' in helpee_conf:
        helpee_conf = helpee_conf.replace('mn-wifi', 'mininet-wifi')
    v2i_bw = get_nodes_v2i_bw(bw_file, run_time, num_nodes, helpee_conf)
    mean_bw = np.mean(v2i_bw)
    all_bw.append(mean_bw)
    num_helpees, disconnect_percentage = \
            get_disconect_duration_in_percentage(helpee_conf, run_time, num_nodes)
    helpee_disconnection.append(disconnect_percentage)
    node_dists = get_node_dists(loc)
    mean_dists = []
    for i in range(num_nodes):
        if i in node_dists:
            dists = node_dists[i][:num_nodes-1]
            mean, std = np.mean(dists), np.std(dists)
            mean_dists.append(mean)
    dist_metric = np.mean(mean_dists)
    all_dist.append(dist_metric)
    return mean_bw, disconnect_percentage, dist_metric
    

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
    
    bw, disconnect, dist = get_setting_characteristic((num_nodes, config["network_trace"], config["location_file"], \
                config["helpee_conf"], config["t"]))
    conf_key = (dir, scheduler, network, mobility, helpee, adaptive, adapt_frame_skipping, bw, disconnect, dist, num_nodes)
    if num_nodes == 8:
        conf_key = (dir, scheduler, network, mobility, helpee, adaptive, adapt_frame_skipping, bw, disconnect, 33, num_nodes)
    return conf_key


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


def get_result_based_on_metric(metric, key_idx):
    result = collections.defaultdict(list)
    result_metric = collections.defaultdict(list)
    for k,v in result_each_run.items():
        m = k[key_idx]
        if metric[0] <= m < metric[1]:
            result_metric['bw'].append(k[-4])
            result_metric['dist'].append(k[-2] if k[-2] < 80 else 80)
            result_metric['disconnect'].append(k[-3])
            result[k[1]].append(v)
    return result, result_metric


def get_rsult_equal_to_node(metric):
    result = collections.defaultdict(list)
    for k,v in result_each_run.items():
        m = k[-1]
        if m == metric:
            result[k[1]].append(v)
    return result


def get_result_based_on_mix_metric(metric):
    result = collections.defaultdict(list)
    for k,v in result_each_run.items():
        m = k[-2] * k[-4]
        if metric[0] <= m < metric[1]:
            result[k[1]].append(v)
    return result


def plot_compare_scheds():
    # bw metric
    bw_threshold = [8, 15]
    for i in range(len(bw_threshold)+1):
        if i == 0:
            metric = [0, bw_threshold[i]]
        elif i > len(bw_threshold)-1:
            metric = [bw_threshold[-1], 400]
        else:
            metric = [bw_threshold[i-1], bw_threshold[i]]
        result = get_result_based_on_metric(metric, -4)[0]
        plot_one_category(result, SCHEDULERS, 'bw', metric)
        
    dist_threshold = [25, 50]
    for i in range(len(dist_threshold)+1):
        if i == 0:
            metric = [0, dist_threshold[i]]
        elif i > len(dist_threshold)-1:
            metric = [dist_threshold[-1], 800]
        else:
            metric = [dist_threshold[i-1], dist_threshold[i]]
        result = get_result_based_on_metric(metric, -2)[0]
        plot_one_category(result, SCHEDULERS, 'distance', metric)
    
    conn_threshold = [30, 35, 40]
    for i in range(len(conn_threshold)+1):
        if i == 0:
            metric = [0, conn_threshold[i]]
        elif i > len(conn_threshold)-1:
            metric = [conn_threshold[-1], 100]
        else:
            metric = [conn_threshold[i-1], conn_threshold[i]]
        result = get_result_based_on_metric(metric, -3)[0]
        plot_one_category(result, SCHEDULERS, 'conn', metric)
    
    if os.path.exists('stats_summary.txt'):
        os.system('rm stats_summary.txt')
    
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(18,5))
    threshold = [30, 50]
    for i in range(len(threshold)+1):
        if i == 0:
            metric = [0, threshold[i]]
        elif i > len(threshold)-1:
            metric = [threshold[-1], 8000]
        else:
            metric = [threshold[i-1], threshold[i]]
        result, rst_metrics = get_result_based_on_metric(metric, -2)
        print(len(result[SCHEDULERS[0]]))
        print(np.mean(rst_metrics['bw']), np.mean(rst_metrics['disconnect']), np.mean(rst_metrics['dist']),
              np.mean(rst_metrics['bw'])* np.mean(rst_metrics['dist']))
        if i == 0:         
            plot_one_category_ax(result, SCHEDULERS, 'distance', metric, axes[1], 'v2i', 1)
        elif i == 1:
            plot_one_category_ax(result, SCHEDULERS, 'distance', metric, axes[0], 'similar', 2)
        else:
            plot_one_category_ax(result, SCHEDULERS, 'distance', metric, axes[i], 'v2v', 3)
        
    axes[1].set_title('b) Similar V2I and V2V Conditions' , y=-0.33)
    axes[1].annotate("Better", fontsize=20, horizontalalignment="center", xy=(0.1, 0.75), xycoords='data',
        xytext=(0.2, 0.72), textcoords='data',
        arrowprops=dict(arrowstyle="->, head_width=0.3", connectionstyle="arc3", lw=3)
        )
    axes[0].set_title('a) Better V2I Conditions', y=-0.33)
    axes[0].annotate("Better", fontsize=20, horizontalalignment="center", xy=(0.1, 0.74), xycoords='data',
        xytext=(0.2, 0.66), textcoords='data',
        arrowprops=dict(arrowstyle="->, head_width=0.3", connectionstyle="arc3", lw=3)
    )
    axes[2].set_title('c) Better V2V Conditions', y=-0.33)
    axes[2].annotate("Better", fontsize=20, horizontalalignment="center", xy=(0.1, 0.65), xycoords='data',
    xytext=(0.2, 0.56), textcoords='data',
    arrowprops=dict(arrowstyle="->, head_width=0.3", connectionstyle="arc3", lw=3)
    )
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('analysis-results/two-dim-acc-latency.pdf')
    
    # node_nums = [12, 14, 16, 18, 20]
    # fig, axes = plt.subplots(1, 2, sharex=True, figsize=(8,3))
    # latency, acc = collections.defaultdict(list), collections.defaultdict(list)
    # latency_err, acc_err = collections.defaultdict(list), collections.defaultdict(list)
    # for num in node_nums:
    #     result = get_rsult_equal_to_node(num)
    #     schedule_data = {}
    #     sched_to_acc = {}
    #     for schedule in SCHEDULERS:
    #         for k, v_list in result.items():
    #             if schedule in k:
    #                 print(schedule, k)
    #                 for v in v_list:
    #                     if schedule not in schedule_data.keys():
    #                         schedule_data[schedule] = v['e2e-latency']
    #                         sched_to_acc[schedule]  = v['detection_acc']
    #                     else:
    #                         schedule_data[schedule] = np.hstack((schedule_data[schedule], v['e2e-latency']))
    #                         sched_to_acc[schedule] = np.hstack((sched_to_acc[schedule],v['detection_acc']))
    #         latency[schedule].append(np.mean(schedule_data[schedule]))
    #         acc[schedule].append(np.mean(sched_to_acc[schedule]))
    #         latency_err[schedule].append(mean_confidence_interval(schedule_data[schedule], 0.99))
    #         acc_err[schedule].append(mean_confidence_interval(sched_to_acc[schedule]))
    #         print(num, schedule, np.mean(schedule_data[schedule]), np.std(schedule_data[schedule]))
    #         print(np.mean(sched_to_acc[schedule]), np.std(sched_to_acc[schedule]))
    # for schedule in SCHEDULERS:  
        
    #     axes[0].errorbar(node_nums, latency[schedule], yerr=latency_err[schedule], capsize=3)
        
    #     if 'no-group' in schedule:
    #         # axes[0].errorbar(node_nums, latency[schedule], yerr=latency_err[schedule], capsize=3)
    #         axes[0].scatter(node_nums, latency[schedule], marker='^',s=20)
    #         axes[1].scatter(node_nums, acc[schedule], marker='^', s=20)
    #         axes[1].errorbar(node_nums, acc[schedule], yerr=acc_err[schedule], capsize=3, label='Harbor\nw/o grouping')
    #     elif 'combined' in schedule:
    #         axes[0].scatter(node_nums, latency[schedule], s=20)
    #         axes[1].scatter(node_nums, acc[schedule], s=20)
    #         axes[1].errorbar(node_nums, acc[schedule], yerr=acc_err[schedule], capsize=3, label='Harbor\nw/ grouping')        
    #     else:
    #         axes[0].scatter(node_nums, latency[schedule], s=20)
    #         axes[1].scatter(node_nums, acc[schedule], s=20)
    #         axes[1].errorbar(node_nums, acc[schedule], yerr=acc_err[schedule], capsize=3, label=schedule)
    # axes[0].set_xticks(node_nums)
    # axes[1].set_xticks(node_nums)
    # axes[0].set_ylabel('Detection Latency (s)')
    # axes[1].set_ylabel('Detection Accuracy')
    # axes[0].grid(linestyle='--')
    # axes[1].grid(linestyle='--')
    # axes[0].set_xlabel('Number of Vehicles')
    # axes[1].set_xlabel('Number of Vehicles')
    # axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()
    # plt.savefig('fig14.pdf')
    
    
def plot_one_category_ax(data, schedules, key, metric, ax, name, fig_num=1):
    schedule_data, schedule_overhead, server_compute_time = {}, {}, {}
    schedule_helpee_data, schedule_helper_data = {}, {}
    sched_to_latency = {}
    sched_to_latency_std = {}
    schedule_to_detected_spaces, sched_to_acc = {}, {}
    summary = {}
    for schedule in schedules:
        for k, v_list in data.items():
            if schedule in k:
                for v in v_list:
                    if schedule not in schedule_data.keys():
                        schedule_overhead[schedule] = v['overhead']
                        server_compute_time[schedule] = v['sched_latency']
                        schedule_data[schedule] = v['all']
                        schedule_helpee_data[schedule] = v['helpee']
                        schedule_helper_data[schedule] = v['helper']
                        schedule_to_detected_spaces[schedule] = [np.mean(v['detected_areas'])]
                        sched_to_acc[schedule] = [np.mean(v['detection_acc'])]
                        sched_to_latency[schedule] = [np.mean(v['e2e-latency'])]
                        sched_to_latency_std[schedule] = [np.std(v['e2e-latency'])]
                    else:
                        schedule_overhead[schedule] = np.hstack((schedule_overhead[schedule], v['overhead']))
                        schedule_data[schedule] = np.hstack((schedule_data[schedule], v['all']))
                        schedule_helpee_data[schedule] = np.hstack((schedule_helpee_data[schedule], v['helpee']))
                        schedule_helper_data[schedule] = np.hstack((schedule_helper_data[schedule], v['helper']))
                        server_compute_time[schedule] = np.hstack((server_compute_time[schedule], v['sched_latency']))
                        schedule_to_detected_spaces[schedule].append(np.mean(v['detected_areas']))
                        sched_to_acc[schedule].append(np.mean(v['detection_acc']))
                        sched_to_latency[schedule].append(np.mean(v['e2e-latency']))
                        sched_to_latency_std[schedule].append(np.std(v['e2e-latency']))
    e2e_summary_dict = {}
    for schedule in schedule_data.keys():
        e2e_summary_dict[schedule] = (np.mean(sched_to_latency[schedule]), np.mean(sched_to_acc[schedule]), np.std(sched_to_latency[schedule]), np.std(sched_to_acc[schedule]))
        summary[schedule] = (np.mean(sched_to_latency[schedule]), np.mean(sched_to_acc[schedule]))
        ax.scatter(np.mean(sched_to_latency[schedule]), np.mean(sched_to_acc[schedule]),
            label='Harbor' if schedule == 'combined-adapt' else schedule, marker=sched_to_marker[schedule],\
            color=sched_to_color[schedule])
        ax.errorbar(np.mean(sched_to_latency[schedule]), np.mean(sched_to_acc[schedule]), \
            xerr=np.std(sched_to_latency[schedule]), yerr=np.std(sched_to_acc[schedule]), capsize=2,
            color=sched_to_color[schedule])
        
        if schedule =='v2i':
            ax.annotate(shced_to_displayed_name[schedule], (summary[schedule][0] + 0.04, summary[schedule][1]+0.005), color=sched_to_color[schedule])
        elif schedule =='v2v-adapt':
            ax.annotate(shced_to_displayed_name[schedule], (summary[schedule][0] - 0.005, summary[schedule][1]-0.014), color=sched_to_color[schedule])
        elif schedule =='v2i-adapt':
            ax.annotate(shced_to_displayed_name[schedule], (summary[schedule][0] - 0.005, summary[schedule][1]+0.001), color=sched_to_color[schedule])
        elif schedule =='v2v':
            ax.annotate(shced_to_displayed_name[schedule], (summary[schedule][0] + 0.045, summary[schedule][1]+0.003), color=sched_to_color[schedule])
        else:
            ax.annotate(shced_to_displayed_name[schedule], (summary[schedule][0] - 0.005, summary[schedule][1] + 0.002), color=sched_to_color[schedule])
        # print("errorbar", np.mean(sched_to_latency_std[schedule]))
        # print(schedule, np.mean(sched_to_latency[schedule]), np.mean(sched_to_acc[schedule]))
    # ax.annotate("Better", fontsize=20, horizontalalignment="center", xy=(0.35, 0.70), xycoords='data',
    #         xytext=(0.4, 0.65), textcoords='data',
    #         arrowprops=dict(arrowstyle="->, head_width=0.3", connectionstyle="arc3", lw=3)
    #         )
    # ax.set_ylim([0.5, 0.85])
    # plt.xlim([0, 0.5])
    import json
    e2e_dict = json.dumps(e2e_summary_dict)
    f = open('%s.json'%name, 'w')
    f.write(e2e_dict)
    f.close()
    ax.set_xlabel("Detection Latency (s)")
    ax.set_ylabel("Detection Accuracy")
    ax.grid(linestyle='--')
    get_comparison_in_summary(summary, fig_num)


def get_comparison_in_summary(summary, fig_num=0):
    summary_file = open('stats_summary.txt', 'a')
    summary_file.write('figure ' + str(fig_num) + '\n')
    for sched, rst in summary.items():
        summary_file.write(sched + '\tlatency_improvement\tacc_improvement' + '\n')
        for comp_sched, comp_rst in summary.items():
            if comp_sched != sched:
                summary_str = sched+'/'+comp_sched + '\t' + str((rst[0]-comp_rst[0])/comp_rst[0]) + '\t' +str(rst[1]-comp_rst[1]) + '\n'
                summary_file.write(summary_str)


def plot_one_category(data, schedules, key, metric):
    schedule_data, schedule_overhead, server_compute_time = {}, {}, {}
    schedule_helpee_data, schedule_helper_data = {}, {}
    sched_to_latency = {}
    sched_to_latency_std = {}
    schedule_to_detected_spaces, sched_to_acc = {}, {}
    for schedule in schedules:
        for k, v_list in data.items():
            if schedule in k:
                for v in v_list:
                    if schedule not in schedule_data.keys():
                        schedule_overhead[schedule] = v['overhead']
                        server_compute_time[schedule] = v['sched_latency']
                        schedule_data[schedule] = v['all']
                        schedule_helpee_data[schedule] = v['helpee']
                        schedule_helper_data[schedule] = v['helper']
                        schedule_to_detected_spaces[schedule] = [np.mean(v['detected_areas'])]
                        sched_to_acc[schedule] = [np.mean(v['detection_acc'])]
                        sched_to_latency[schedule] = [np.mean(v['e2e-latency'])]
                        sched_to_latency_std[schedule] = [np.std(v['e2e-latency'])]
                    else:
                        schedule_overhead[schedule] = np.hstack((schedule_overhead[schedule], v['overhead']))
                        schedule_data[schedule] = np.hstack((schedule_data[schedule], v['all']))
                        schedule_helpee_data[schedule] = np.hstack((schedule_helpee_data[schedule], v['helpee']))
                        schedule_helper_data[schedule] = np.hstack((schedule_helper_data[schedule], v['helper']))
                        server_compute_time[schedule] = np.hstack((server_compute_time[schedule], v['sched_latency']))
                        schedule_to_detected_spaces[schedule].append(np.mean(v['detected_areas']))
                        sched_to_acc[schedule].append(np.mean(v['detection_acc']))
                        sched_to_latency[schedule].append(np.mean(v['e2e-latency']))
                        sched_to_latency_std[schedule].append(np.std(v['e2e-latency']))
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    for schedule in schedule_data.keys():
        ax.scatter(np.mean(sched_to_latency[schedule]), np.mean(sched_to_acc[schedule]),
            label='Harbor' if schedule == 'combined-adapt' else schedule, marker=sched_to_marker[schedule],\
            color=sched_to_color[schedule])
        ax.errorbar(np.mean(sched_to_latency[schedule]), np.mean(sched_to_acc[schedule]), \
            xerr=np.std(sched_to_latency[schedule]), yerr=np.std(sched_to_acc[schedule]), capsize=2,
            color=sched_to_color[schedule])
        # print("errorbar", np.mean(sched_to_latency_std[schedule]))
        # print(schedule, np.mean(sched_to_latency[schedule]), np.mean(sched_to_acc[schedule]))
    # ax.annotate("Better", fontsize=20, horizontalalignment="center", xy=(0.15, 0.76), xycoords='data',
    #         xytext=(0.1, 0.83), textcoords='data',
    #         arrowprops=dict(arrowstyle="<-, head_width=0.3", connectionstyle="arc3", lw=3)
    #         )
    # plt.ylim([0.5, 0.9])
    # plt.xlim([0, 0.5])
    plt.xlabel("Detection Latency (s)")
    plt.ylabel("Detection Accuracy")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis-results/two-dim-acc-latency-%s-%d-%d.png'%(key, metric[0], metric[1]))

        

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

def plot_based_on_setting(num_nodes):
    all_keys = generate_keys(LOC, BW, HELPEE, SCHEDULERS)

    # result = construct_result_based_on_keys(all_keys)
    result_all_frame = construct_frame_result_based_on_keys(all_keys)
    # print(result)
    for loc in LOC:
        for bw in BW:
            for helpee in HELPEE:
                # partial_results = find_data_with_partial_keys((loc, bw, helpee), result)
                # plot_dict_data_box(partial_results, str([loc, bw, helpee]), 2)
                # plot_dict_data_cdf(partial_results, str([loc, bw, helpee]), 2)
                # partial_results = find_data_with_partial_keys((loc, bw, helpee), result_all_frame)
                # plot_full_frame(partial_results, str([loc, bw, helpee]), 2)
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default="~/v2x/", type=str, help="data directory")
    parser.add_argument('-p', '--prefix', default='data-', type=str, help='prefix on data dir to analyze')
    args = parser.parse_args()
    data_dir = args.data_dir
    key = args.prefix
    os.system('mkdir %s/analysis-results/'%data_dir)
    get_all_runs_results(data_dir, key)
    plot_compare_scheds()
    # print(result_each_run)
    


if __name__ == '__main__':
    main()
