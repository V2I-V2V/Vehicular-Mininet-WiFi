import os, sys
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_experiment
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns

font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)

result_each_run = {}
result_per_node = {}
num_nodes = 6

SCHEDULER = []
LOC = []
BW = []
HELPEE = []


def generate_keys(locs, bws, schedulers=None):
    keys = []
    for l in locs:
        for b in bws:
            if schedulers is not None:
                for s in schedulers:
                    keys.append((l,b,s))
            else:
                keys.append((l,b))
    return keys


def check_keys_matched(key, file_key):
    # file_key = file_key[14:]
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

# Create a setting

def find_data_with_partial_keys(partial_keys, data):
    result = {}
    for k,v in data.items():
        matched = check_keys_matched(partial_keys, k)
        if matched:
            result[k] = v
    return result

def plot_dict_data_box(dict, name):
    plt.figure()
    labels, data = dict.keys(), dict.values()
    ticks = []
    for label in labels:
        ticks.append(label[2])
    # print('Boxplot data')
    # print(data)
    x_positions = np.arange(len(result_each_run.keys()))
    cnt = 0
    for k in labels:
        plt.boxplot(dict[k], positions=np.array([x_positions[cnt]]), whis=(5, 95), autorange=True, showfliers=False) # TODO: select part of the tail
        cnt += 1
    plt.xticks(range(0, len(ticks)), ticks, fontsize=10)
    plt.ylabel('Latency (s)')
    plt.savefig('analysis-results/%s.png'%name)

def plot_dict_data_cdf(dict, name):
    plt.figure()
    labels, data = list(dict.keys()), list(dict.values())
    print(labels)
    ticks = []
    for label_idx in range(len(labels)):
        ticks.append(labels[label_idx][2])
        sns.ecdfplot(data[label_idx], label=labels[label_idx][2])
    plt.legend()
    plt.savefig('analysis-results/%s-cdf.png'%name)


def plot_based_on_setting():
    all_keys = generate_keys(LOC, BW, SCHEDULER)

    result = construct_result_based_on_keys(all_keys)
    # print(result)
    for loc in LOC:
        for bw in BW:
            partial_results = find_data_with_partial_keys([loc, bw], result)
            plot_dict_data_box(partial_results, str([loc, bw]))
            plot_dict_data_cdf(partial_results, str([loc, bw]))
            



def get_per_experiment_stats(result_dir, node_num):
    stats = {}
    for i in range(node_num):
        stats[i] = np.loadtxt(result_dir+'/node%d_delay.txt'%i)
    stats['all'] = np.loadtxt(result_dir+'/all_delay.txt')
    return stats


def get_all_runs_results(analyze_type, data_dir, frames, key):
    global num_nodes
    if analyze_type == 'single':
        config = run_experiment.parse_config_from_file(data_dir+'/config.txt')
        # frames = int(config[""])
        num_nodes = int(config["num_of_nodes"])
        os.system('sudo python3 %s/calc_delay.py %d %s %d'%(CODE_DIR, num_nodes, data_dir, frames))
    elif analyze_type == 'multi':
        dirs = os.listdir(data_dir)
        for dir in dirs:
            if key in dir:
                config = run_experiment.parse_config_from_file(data_dir+dir+'/config.txt')
                scheduler = config["scheduler"]
                network = config["network_trace"].split('/')[-1][:-4]
                mobility = config["location_file"].split('/')[-1][:-4]
                if scheduler not in SCHEDULER:
                    SCHEDULER.append(scheduler)
                if network not in BW:
                    BW.append(network)
                if mobility not in LOC:
                    LOC.append(mobility)
                conf_key = (dir, scheduler, network, mobility)
                num_nodes = int(config["num_of_nodes"])
                files = os.listdir(data_dir+dir)
                if 'all_delay.txt' not in files:           
                    os.system('sudo python3 %s/calc_delay.py %s/ %d %d'%(CODE_DIR, dir, num_nodes, frames))
                result_each_run[conf_key] = get_per_experiment_stats(data_dir+dir, num_nodes)


def plot_bar_across_runs():
    fig = plt.figure()
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


def plot_bar_compare_schedule(schedules):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_positions = np.arange(len(result_each_run.keys()))
    cnt = 0
    schedule_data = {}
    for schedule in schedules:
        for k, v in result_each_run.items():
            if schedule in k:
                if schedule not in schedule_data.keys():
                    schedule_data[schedule] = v['all']
                else:
                    schedule_data[schedule] = np.hstack((schedule_data[schedule], v['all']))
    
    for schedule in schedule_data.keys():
        ax.boxplot(schedule_data[schedule], positions=np.array([x_positions[cnt]]), whis=(5, 95), autorange=True, showfliers=False)
        cnt += 1

    ax.set_xticklabels(schedules)
    plt.ylabel('Latency (s)')
    plt.savefig('analysis-results/schedule_compare.png')


def calculate_per_node_mean(setting):
    print("Setting: %s" % str(setting))
    node_result = {}
    for k, v in result_each_run.items():
        matched = check_keys_matched(setting, k)
        if matched:
            print(k)
            for i in range(num_nodes):
                node_latency = v[i] 
                if i in node_result.keys():
                    node_result[i].append(np.mean(node_latency))
                else:
                    node_result[i] = [np.mean(node_latency)]
   
    # print(node_result)
    return node_result


def plot_bar(data, name):
    if len(data.values()) == 0:
        return None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(data.values()))
    values = data.values()
    cnt = 0
    for value in values:
        mean, std = np.mean(value), np.std(value)
        x = np.arange(cnt, (cnt+1))
        plt.errorbar(x, mean, yerr=std, capsize=4)
        ax.scatter(x, mean)
        cnt += 1
    
    plt.ylabel('Average Latency (s)')
    plt.savefig('analysis-results/%s-per-node.png'%name)

def repeat_exp_analysis():
    all_keys = generate_keys(LOC, BW, SCHEDULER)
    for key in all_keys:
        node_result = calculate_per_node_mean(key)
        # print("node results")
        # print(node_result)
        plot_bar(node_result, str(key))

        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default="single", type=str, help="analyze type")
    parser.add_argument('-d', '--data_dir', default="~/v2x/", type=str, help="data directory")
    # parser.add_argument('-n', '--num_nodes', default=6, type=int, help="number of nodes")
    parser.add_argument('-f', '--frames', default=80, type=int, help='number of frames considered')
    parser.add_argument('-k', '--keys', default='data-', type=str, help='key on data dir')

    args = parser.parse_args()

    analyze_type = args.type
    data_dir = args.data_dir
    frames = args.frames
    key = args.keys

    os.system('mkdir analysis-results/')

    get_all_runs_results(analyze_type, data_dir, frames, key)

    plot_bar_across_runs()

    for sched in SCHEDULER:
        plot_bar_based_on_schedule(sched)
    
    plot_bar_compare_schedule(SCHEDULER)

    plot_based_on_setting()

    # calculate_per_node_std()
    repeat_exp_analysis()


if __name__ == '__main__':
    main()