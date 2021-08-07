import sys, os
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15}

matplotlib.rc('font', **font)

def plot_trajectory(loc_file, save_dir='./'):
    trajs = {}
    traj = np.loadtxt(loc_file)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)

    for i in range(int(traj.shape[1]/2)):
        ax.plot(traj[:,2*i], traj[:,2*i+1], label='node%d'%i)
        trajs[i] = traj[:,2*i:2*i+2]
    for i in range(int(traj.shape[1]/2)):
        ax.scatter([traj[:,2*i][0]], [traj[:,2*i+1][0]], s=34)

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir + 'traj.png')
    plot_distance_for_each_node(trajs, save_dir)


def plot_distance_for_each_node(trajs, save_dir='./'):
    fig = plt.figure(figsize=(7,20))
    cnt = 1
    for start_node in trajs.keys():
        ax = fig.add_subplot(len(trajs.keys()),1,cnt)
        for end_node in trajs.keys():
            if start_node != end_node:
                start_node_traj, end_node_traj = trajs[start_node], trajs[end_node]
                dist = np.linalg.norm(start_node_traj - end_node_traj, axis=1)
                if len(dist) > 10:
                    ax.plot(np.arange(0, int(len(dist)/10), 0.1), dist, label='dist to %d'%end_node)
                else:
                    ax.plot(np.arange(0, len(dist)), dist, label='dist to %d'%end_node)
                # ax.set_xticks(np.arange(0, math))
                # if len(dist) > 10:
                #     ax.set_xticks(np.arange(0, len(dist)/10))
                ax.set_ylabel('distance (m)')
                ax.set_xlabel('time (s)')
                ax.legend()
        cnt += 1
        
    # plt.ylabel('y (m)')
    plt.tight_layout()
    plt.savefig(save_dir + 'node-distances.png')

def plot_loc_at_timestamp(loc_file, timestamp, save_dir='./'):
    traj = np.loadtxt(loc_file)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
        
    time_idx_in_100ms = int(timestamp * 10)
    if time_idx_in_100ms > traj.shape[0]:
        loc = traj[-1]
    else:
        loc = traj[time_idx_in_100ms]
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    for i in range(int(len(loc)/2)):
        ax.scatter([loc[2*i]], [loc[2*i+1]], s=34) # , label='node%d'%i
        ax.annotate('%d'%i, xy=(loc[2*i], loc[2*i+1]+10))
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir + 'loc-at-%ds.png'%timestamp)
    


if __name__ == '__main__':
    if len(sys.argv) == 2:
        plot_trajectory(sys.argv[1])
    elif len(sys.argv) == 3:
        plot_loc_at_timestamp(sys.argv[1], float(sys.argv[2]))
    else:
        print("To plot trajectory, use python3 trajectory.py <path_to_loc_trace>, \
            to plot location at cerntain timestamp, use python trajectory.py \
                <path_to_loc_trace> <time in seconds>")
        sys.exit(1)
    
    

    # fig = plt.figure(figsize=(7,7))
    # ax = fig.add_subplot(111)

    # for i in range(int(traj.shape[1]/2)):
    #     ax.scatter([traj[:,2*i][-1]], [traj[:,2*i+1][-1]], s=184, label='node%d'%i)
    #     print(traj[:,2*i+1][-1])
    #     trajs[i] = traj[:,2*i:2*i+2]

    # # plt.xlim([-10, 50])
    # # plt.ylim([360, 500])
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.legend()
    # plt.tight_layout()

    # plt.savefig('traj-end.png')

# for k in trajs.keys():
#     for k in trajs
# dist = np.linalg.norm(trajs[2] - trajs[1], axis=1)
# print(np.max(dist))
# dist = np.linalg.norm(trajs[1] - trajs[4], axis=1)
# print(np.max(dist))
# dist = np.linalg.norm(trajs[1] - trajs[3], axis=1)
# print(dist)
# dist = np.linalg.norm(trajs[1] - trajs[5], axis=1)
# print(dist)
# dist = np.linalg.norm(trajs[1] - trajs[0], axis=1)
# print(dist)


