import os
import sys
import numpy as np
from numpy.core.fromnumeric import argmax
import ptcl_utils
import matplotlib.pyplot as plt
import matplotlib
import collections
import argparse
matplotlib.rc('font', **{'size': 18})
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pointcloud import dracoDecode, dracoEncode

# ptcl = ptcl_utils.read_ptcl_data('/home/ryanzhu/MultiVehiclePerception/86_130_merged_10_8.bin')
# ptcl_s1 = ptcl_utils.read_ptcl_data('/home/ryanzhu/V2I+V2V/Carla/lidar/86/1000.npy')
# ptcl_s2 = ptcl_utils.read_ptcl_data('/home/ryanzhu/V2I+V2V/Carla/lidar/174/1000.npy')
#
# locs = [(-21.642448, 193.882751), (-9.493179, 114.654312), (-5.69124699, 183.77157593)]
#
ranges = [5, 10, 15, 30, 50, 75, 100]
ransac_t = 0.08
#
# ptcl_s1 = ptcl_utils.get_points_within_center(ptcl_s1, space_range=100)
# ptcl_s2 = ptcl_utils.get_points_within_center(ptcl_s2)
# density_s1 = ptcl_utils.avg_point_density_in_range(ptcl_s1, ranges, space_range=100)
# density_s2 = ptcl_utils.avg_point_density_in_range(ptcl_s2, ranges, space_range=100)
#
# filter_merged_ptcls1 = ptcl_utils.get_points_within_center(ptcl, space_range=100, center=locs[0])
# filter_merged_ptcls2 = ptcl_utils.get_points_within_center(ptcl, space_range=100, center=locs[2])
#
# density_s1_merged = ptcl_utils.avg_point_density_in_range(filter_merged_ptcls1, ranges, space_range=100)
# density_s2_merged = ptcl_utils.avg_point_density_in_range(filter_merged_ptcls2, ranges, space_range=100)
#
# print(density_s1)
# print(density_s1_merged)
# print(density_s2)
# print(density_s2_merged)
#
#
# plt.plot(ranges[2:], density_s1[2:-1], label='single vehicle')
# plt.plot(ranges[2:], density_s1_merged[2:-1], label='merged')
# plt.xticks(ranges[2:], ['<15', '15-30', '30-50', '50-75', '75-100'])
#
#
# plt.ylabel('Points per square meter')
# plt.xlabel('range (m)')
# plt.legend()
# plt.show()


def get_pointclouds(args):
    vehicle_id_to_ptcl_density_raw = collections.defaultdict(list)
    vehicle_id_to_ptcl_density_merged = collections.defaultdict(list)
    # provided dataset directory and frame id
    if args.datadir != "":
        for f_id in range(1000, 1001):
            # load pointclouds
            lidar_dir = os.path.join(args.datadir, "lidar")
            vehicle_positions = {}
            vehicle_prediction = {}
            pointclouds, vehicle_ids = [], []
            for vehicle_id in ['86', '130']: # os.listdir(lidar_dir)
                pcd_file = os.path.join(lidar_dir, vehicle_id, str(f_id))
                if os.path.isfile(pcd_file + ".npy"):
                    pcd = np.load(pcd_file + ".npy")
                elif os.path.isfile(pcd_file + ".bin"):
                    pcd = np.fromfile(pcd_file + ".bin", dtype=np.float32, count=-1).reshape(-1, 4)
                else:
                    pcd = None

                if pcd is not None:
                    pcd[:, 3] = 1
                    density_raw = ptcl_utils.avg_point_density_in_range(pcd, ranges, space_range=100)
                    pcd_raw = np.copy(pcd)
                    vehicle_id_to_ptcl_density_raw[vehicle_id].append(density_raw)
                    encoded, _ = dracoEncode(pcd, 10, 6)
                    decoded = dracoDecode(encoded)
                    pcd = np.concatenate([decoded, np.ones((decoded.shape[0], 1))], axis=1)
                    if os.path.isfile(pcd_file + ".trans.npy"):
                        trans = np.load(pcd_file + ".trans.npy")
                        pcd[:, 3] = 1
                        pcd = np.dot(trans, pcd[:, :4].T).T
                        pcd_raw = np.dot(pcd_raw[:, :4], trans.T)
                        dummy = np.zeros((4, 1))
                        dummy[3] = 1
                        new_pos = np.dot(trans, dummy).T
                    vehicle_prediction[vehicle_id] = ptcl_utils.ransac_predict(pcd_raw, threshold=ransac_t)
                    # np.save('%s_%d_pred.npy'%(vehicle_id, f_id), vehicle_prediction[vehicle_id])
                    pointclouds.append(pcd)
                    # print(new_pos.shape)
                    vehicle_positions[vehicle_id] = (new_pos[0,0], new_pos[0,1])
                    # print(vehicle_positions[vehicle_id])
                    vehicle_ids.append(vehicle_id)
            merged_gt_file = '_'.join(vehicle_ids) + '_gt_' + str(f_id)+'.npy'                     
            merged = np.vstack(pointclouds)
            if os.path.exists(merged_gt_file):
                gt_ptcl = ptcl_utils.read_ptcl_data(merged_gt_file)
                merged_pred = ptcl_utils.ransac_predict(merged, threshold=ransac_t)
                np.save('%s_%d_pred.npy'%('_'.join(vehicle_ids), f_id), merged_pred)
                s_acc, r_acc, m_acc, o_acc = [], [], [], []
                for v_id in vehicle_ids:
                    gt_grid, _ = ptcl_utils.calculate_grid_label_ransac(1, gt_ptcl, center=vehicle_positions[v_id])
                    single_pred_grid, _ = ptcl_utils.calculate_grid_label_ransac(1, vehicle_prediction[vehicle_id], center=vehicle_positions[v_id])
                    merged_pred_grid, _ = ptcl_utils.calculate_grid_label_ransac(1, merged_pred, center=vehicle_positions[v_id])
                    single_acc, _ = ptcl_utils.calculate_precision(single_pred_grid, gt_grid)
                    merged_acc, _ = ptcl_utils.calculate_precision(merged_pred_grid, gt_grid)
                    updated_grid = ptcl_utils.combine_merged_results(single_pred_grid, merged_pred_grid)
                    update_acc, _ = ptcl_utils.calculate_precision(updated_grid, gt_grid)
                    oracle_acc = ptcl_utils.calculate_oracle_accuracy(single_pred_grid, merged_pred_grid, gt_grid)
                    ranges_acc = []
                    for r in ranges:
                        combine_grid = ptcl_utils.combine_merged_results_on_remote(single_pred_grid, merged_pred_grid, threshold=r)
                        combine_acc, _ = ptcl_utils.calculate_precision(combine_grid, gt_grid)
                        ranges_acc.append(combine_acc)
                    # ptcl_utils.plot_grid(gt_grid, 1)
                    # ptcl_utils.plot_grid(merged_pred_grid, 1)
                    # ptcl_utils.plot_grid(single_pred_grid, 1)
                    s_acc.append(single_acc)
                    r_acc.append(merged_acc)
                    m_acc.append(max(ranges_acc))
                    o_acc.append(oracle_acc)
                    print('Vehicle ID:', v_id, single_acc, merged_acc, update_acc)
                    print('Best acc', max(ranges_acc), ranges[argmax(ranges_acc)])
                    # print(ranges_acc)
                print(np.mean(s_acc), np.mean(r_acc), np.mean(m_acc), np.mean(o_acc))
            # print(merged)
            
            for v_id in vehicle_ids:
                filter_merged_ptcl = ptcl_utils.get_points_within_center(merged, space_range=100, center=vehicle_positions[v_id])
                density_merged = ptcl_utils.avg_point_density_in_range(filter_merged_ptcl, ranges, space_range=100)
                # print(density_merged)
                vehicle_id_to_ptcl_density_merged[v_id].append(density_merged)
                print(np.array(density_merged)/np.array(vehicle_id_to_ptcl_density_raw[v_id]))
            

        # print(np.array(vehicle_id_to_ptcl_density_raw['86']).shape)
        # print(np.array(vehicle_id_to_ptcl_density_merged['86']).shape)
        
        for v_id in vehicle_ids:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            raw_density = np.array(vehicle_id_to_ptcl_density_raw[v_id])
            merged_density = np.array(vehicle_id_to_ptcl_density_merged[v_id])
            # print('merged density', )
            single_mean = raw_density.mean(axis=0)
            single_var = raw_density.var(axis=0)
            merged_mean = merged_density.mean(axis=0)
            merged_var = merged_density.var(axis=0)
            
            ret = ax.errorbar(np.arange(len(single_mean)), single_mean, yerr=single_var, capsize=4, label='raw', ls='--')
            ax.errorbar(np.arange(len(merged_mean)), merged_mean, yerr=merged_var, capsize=4, label='compression-high', ls='-.')
            print(merged_mean, merged_var)
            # data = [173.69533869, 220.37654453, 46.34846591, 9.59362418, 3.48429959, 3.33150766, 6.52065987]
            # ax.errorbar(np.arange(len(merged_mean)), data, yerr=merged_var, capsize=4, label='compression-low', alpha=0.5)
            ax.set_xticks(np.arange(len(single_mean)))
            ax.set_xticklabels(['<5', '5-10', '10-15', '15-30', '30-50', '50-75', '75-100'])
            plt.legend()
            plt.ylabel('Points per square meter')
            plt.xlabel('range (m)')   
            plt.savefig('%s-density.pdf'%v_id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to visualize point clouds")
    parser.add_argument("--datadir", type=str, default="", help="Path to the dataset folder")
    parser.add_argument("--frame", type=int, default=-1, help="Frame id (must specify datadir)")
    args = parser.parse_args()
    get_pointclouds(args)