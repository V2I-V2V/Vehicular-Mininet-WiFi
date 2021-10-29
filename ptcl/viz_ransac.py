import os
import sys
import numpy as np
import argparse
import ptcl_utils
from ground import my_ransac_v5


def main(args):
    positions = []
    vehicle_ids = []
    pointclouds = []
    space_detected_single = {}
    space_detected_multi = {}

    if args.datadir != "" and args.frame >= 0:
        # load pointclouds
        lidar_dir = os.path.join(args.datadir, "lidar")
        label_dir = os.path.join(args.datadir, 'labels')
        for vehicle_id in os.listdir(lidar_dir):
            pcd_file = os.path.join(lidar_dir, vehicle_id, str(args.frame))
            if os.path.isfile(pcd_file + ".npy"):
                pcd = np.load(pcd_file + ".npy")
            elif os.path.isfile(pcd_file + ".bin"):
                pcd = np.fromfile(pcd_file + ".bin", dtype=np.float32, count=-1).reshape(-1, 4)
            else:
                pcd = None
            p2, p1, best_model, _ = my_ransac_v5(pcd, 10000, P=0.8, distance_threshold=0.35,
                                                 lidar_height=-2.03727 + 0.1, lidar_height_down=-2.03727 - 0.1,
                                                 use_all_sample=True)
            if os.path.isfile(pcd_file + ".trans.npy"):
                trans = np.load(pcd_file + ".trans.npy")
                pcd[:, 3] = 1
                ptcl = np.dot(trans, pcd[:, :4].T).T

            # pcl, P=0.8, distance_threshold=0.15, lidar_height=-h+0.05 0.15, lidar_height_down=-h-0.15 -0.2
            pcd[:, 3] = 0    # object
            pcd[p2, 3] = 1   # ground
            np.save(pcd_file + '.npy', pcd)
            p2, p1, best_model, _ = my_ransac_v5(ptcl, 10000, P=0.8, distance_threshold=0.35,
                                                 lidar_height=-2.03727 + 0.1, lidar_height_down=-2.03727 - 0.1,
                                                 use_all_sample=True)
            ptcl[:, 3] = 0  # object
            ptcl[p2, 3] = 1  # ground
            np.save('%d_%s_pred.npy' % (args.frame, vehicle_id), ptcl)
            loc = tuple(np.loadtxt(pcd_file + '.txt')[:2])
            grid = ptcl_utils.calculate_grid_label_ransac(1, pcd, 100)
            print(vehicle_id, loc, len(grid[grid != 0]))
            ptcl_utils.plot_grid(grid, 1)
            space_detected_single[vehicle_id] = len(grid[grid != 0])
            pointclouds.append(pcd)
            vehicle_ids.append(vehicle_id)
            positions.append(loc)

        lidar_merged_dir = os.path.join(args.datadir, "merged")
        merged_detect_pcd_file = os.path.join(lidar_merged_dir, str(args.frame))
        merged_pcd = ptcl_utils.read_ptcl_data(merged_detect_pcd_file+'.bin')
        for i in range(len(vehicle_ids)):
            loc = positions[i]
            grid = ptcl_utils.calculate_grid_label_ransac(1, merged_pcd, 100, center=loc)
            print(vehicle_ids[i], loc, len(grid[grid != 0]), len(grid[grid != 0])/space_detected_single[vehicle_ids[i]]-1)
        ptcl_utils.plot_vehicle_location(positions, vehicle_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to calculate drivable space for point clouds")
    parser.add_argument("--datadir", type=str, default="", help="Path to the dataset folder")
    parser.add_argument("--frame", type=int, default=-1, help="Frame id (must specify datadir)")
    args = parser.parse_args()
    main(args)
