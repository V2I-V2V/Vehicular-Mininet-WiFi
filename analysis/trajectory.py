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
        trajs[i] = traj[:,2*i:2*i+2]

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir + 'traj.png')


if __name__ == '__main__':
    plot_trajectory(sys.argv[1])

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


