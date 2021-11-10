import collections
import json
import os
import sys
import argparse
import numpy as np
import ptcl_utils
import matplotlib.pyplot as plt
import matplotlib
from ground import my_ransac_v5
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointcloud

matplotlib.rc('font', **{'size': 18})

if __name__ == '__main__':
    # datadir = '/home/ryanzhu/V2I+V2V/carla-bin/lidar/119/labels'
    # for label in os.listdir(datadir):
    #     lab = np.fromfile(os.path.join(datadir, label), dtype=np.uint32)
    #     lab[lab == 31] = 40
    #     with open(os.path.join(datadir, label), 'w') as f:
    #         lab.tofile(f)

    qb_range = np.arange(12, 6, -1, dtype=int)
    data_dir = '/home/ryanzhu/V2I+V2V/carla-bin/lidar/119/'

    ptcl_dir = data_dir + 'velodyne/'
    label_dir = data_dir + 'labels/'
    qb_to_acc = collections.defaultdict(list)
    for ptcl_name in os.listdir(ptcl_dir):
        ptcl = ptcl_utils.read_ptcl_data(ptcl_dir + ptcl_name)
        label = np.fromfile(label_dir + ptcl_name[:-4] + '.label', dtype=np.uint32)
        GndSeg = ptcl_utils.get_GndSeg(label, [40])
        ptcl[:, 3] = GndSeg
        gt_label, _ = ptcl_utils.calculate_grid_label_ransac(1, ptcl)
        for qb in qb_range:
            encoded, ratio = pointcloud.dracoEncode(ptcl, 10, qb)
            decoded = pointcloud.dracoDecode(encoded)
            decoded = np.hstack([decoded, np.ones((ptcl.shape[0], 1), dtype=np.float32)])
            p2, p1, best_model, _ = my_ransac_v5(decoded, 10000, P=0.8, distance_threshold=0.10,
                                                 lidar_height=-2.03727 + 0.1, lidar_height_down=-2.03727 - 0.1,
                                                 use_all_sample=True)
            decoded[:, 3] = 0  # object
            decoded[p2, 3] = 1  # ground
            decoded_grid, _ = ptcl_utils.calculate_grid_label_ransac(1, decoded)
            acc, _ = ptcl_utils.calculate_precision(decoded_grid, gt_label)
            print(qb, acc)
            qb_to_acc[qb].append(acc)

    import json

    # print(qb_to_acc)
    # acc_json = json.dumps(qb_to_acc)
    # f = open("acc_dict_119.json", "w")
    # f.write(acc_json)
    # f.close()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    cnt = 0
    # ax.boxplot(qb_to_size_dict['raw'], positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
    # cnt += 1
    for qb, sizes in qb_to_acc.items():
        if qb != 'raw':
            ax.boxplot(sizes, positions=np.array([cnt]), whis=(5, 95), autorange=True, showfliers=False)
            cnt += 1
    tick_labels = []
    for qb in qb_range:
        tick_labels.append(str(qb))
    ax.set_xticks(np.arange(len(qb_to_acc.keys())))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('Drivable Space\nDetection Accuracy')
    ax.set_xlabel('Compression Level (qb)')
    plt.tight_layout()
    plt.savefig('acc_vs_compression_qb.pdf')
    # ptcl = ptcl_utils.read_ptcl_data()
