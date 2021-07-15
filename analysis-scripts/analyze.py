import os, sys
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_experiment
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)

result_each_run = {}
result_per_node = {}
num_nodes = 6


def get_per_experiment_stats(result_dir, node_num):
    stats = {}
    for i in range(node_num):
        stats[i] = np.loadtxt(result_dir+'/node%d_delay.txt'%i)
    stats['all'] = np.loadtxt(result_dir+'/all_delay.txt')
    return stats


# def check_if_result_generated(dir):

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
                num_nodes = int(config["num_of_nodes"])
                files = os.listdir(data_dir+dir)
                if 'all_delay.txt' not in files:           
                    os.system('sudo python3 %s/calc_delay.py %s/ %d %d'%(CODE_DIR, dir, num_nodes, frames))
                result_each_run[dir] = get_per_experiment_stats(data_dir+dir, num_nodes)


def plot_bar_across_runs():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_positions = np.arange(len(result_each_run.keys()))
    cnt = 0
    for k, v in result_each_run.items():
        print(v['all'])
        ax.boxplot(v['all'], positions=np.array([x_positions[cnt]]), autorange=True, showfliers=False)
        cnt += 1
    plt.show()


def calculate_per_node_std():
    node_result = {}
    for i in range(num_nodes):
        node_result[i] = []
        for k, v in result_each_run.items():
            node_latency = v[i]    
            node_result[i].append(np.std(node_latency))
    print(node_result)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default="single", type=str, help="analyze type")
    parser.add_argument('-d', '--data_dir', default="~/v2x/", type=str, help="data directory")
    # parser.add_argument('-n', '--num_nodes', default=6, type=int, help="number of nodes")
    parser.add_argument('-f', '--frames', default=80, type=int, help='number of frames considered')
    parser.add_argument('-k', '--keys', default='data-', type=str, help='key on data dir')

    args = parser.parse_args()

    analyze_type = args.type
    # num_nodes = args.num_nodes
    data_dir = args.data_dir
    frames = args.frames
    key = args.keys

    get_all_runs_results(analyze_type, data_dir, frames, key)

    plot_bar_across_runs()

    calculate_per_node_std()


if __name__ == '__main__':
    main()