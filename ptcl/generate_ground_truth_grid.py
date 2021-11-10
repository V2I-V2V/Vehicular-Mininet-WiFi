import os
import sys
import argparse
import numpy as np
import open3d as o3d
import ptcl_utils


def rotation_matrix(pitch, yaw, roll):
    R = np.array([[np.cos(yaw) * np.cos(pitch),
                   np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
                   np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
                  [np.sin(yaw) * np.cos(pitch),
                   np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
                   np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
                  [-np.sin(pitch),
                   np.cos(pitch) * np.sin(roll),
                   np.cos(pitch) * np.cos(roll)]])
    return R


def draw_open3d(pointclouds, labels, show=True, save=""):
    # TODO: paint different pcd in various colors?
    # pointcloud_all = o3d.geometry.PointCloud()
    # pointcloud_all.points = o3d.utility.Vector3dVector(np.vstack(pointclouds)[:,:3])
    # pointcloud_all.paint_uniform_color([0, 0, 1])
    # merged = np.vstack(pointclouds)
    # with open('86_174_merged_10_8.bin', 'w') as f:
    #     merged = merged.astype(np.float32)
    #     merged.tofile(f)

    pcds = []
    colors = [[0, 0.4, 1], [0, 1, 0.3], [0.0, 0.4, 0.5], [0.1, 0.7, 0.4], [0, 0.8, 0.5],
              [0.9, 1, 1], [1, 1, 0], [1, 0.5, 0.5], [1, 0, 0.5], [1, 0, 1]]  # blue, green, orange, red
    for pcl in pointclouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
        pcd.paint_uniform_color(colors.pop(0))
        pcds.append(pcd)

    bboxes = []
    location = labels[:, 3:6]
    extent = labels[:, 6:9] * 2
    rotation = labels[:, 9:12] * np.pi / 180
    R = rotation_matrix(rotation[:, 0], rotation[:, 1], rotation[:, 2])
    for i in range(labels.shape[0]):
        bbox = o3d.geometry.OrientedBoundingBox(center=location[i, :], R=R[:, :, i], extent=extent[i, :])
        bbox.color = [1, 0, 0]
        bboxes.append(bbox)

    o3d.visualization.draw_geometries(bboxes + pcds)


def get_number_of_points_in_region(ptcl, x_center, y_center, x_width, y_width):
    filtered = ptcl[ptcl[:, 0] > x_center - x_width / 2]
    filtered = filtered[filtered[:, 0] < x_center + x_width / 2]
    filtered = filtered[filtered[:, 1] > y_center - y_width / 2]
    filtered = filtered[filtered[:, 1] < y_center + y_width / 2]
    return filtered.shape[0]


def main(args):
    region_points = {}
    label_data = None

    # provided dataset directory and frame id
    if args.datadir != "" and args.frame >= 0:

        # load pointclouds
        lidar_dir = args.datadir
        label_dir = '/home/ryanzhu/Downloads/Carla-Labels/'
        for f_id in range(1000, 1080):
            vehicle_ids = []
            pointclouds = []
            vehicle_positions = {}
            for vehicle_id in ['86', '97', '108', '119', '163']:
                pcd_file = os.path.join(lidar_dir, vehicle_id, str(f_id))
                if os.path.isfile(pcd_file + ".npy"):
                    pcd = np.load(pcd_file + ".npy")
                elif os.path.isfile(pcd_file + ".bin"):
                    pcd = np.fromfile(pcd_file + ".bin", dtype=np.float32, count=-1).reshape(-1, 4)
                else:
                    pcd = None

                if pcd is not None:
                    pcd[:, 3] = 1
                    if os.path.isfile(pcd_file + ".trans.npy"):
                        trans = np.load(pcd_file + ".trans.npy")
                        pcd[:, 3] = 1
                        pcd = np.dot(trans, pcd[:, :4].T).T
                        dummy = np.zeros((4, 1))
                        dummy[3] = 1
                        new_pos = np.dot(trans, dummy).T
                        print(new_pos)
                        np.savetxt(pcd_file + '.txt', new_pos, fmt='%f')
                        vehicle_positions[vehicle_id] = (new_pos[0, 0], new_pos[0, 1])
                    # load labels
                    label_f = os.path.join(label_dir, vehicle_id, 'labels', str(f_id) + '.label')
                    label_index = np.fromfile(label_f, dtype=np.uint32)
                    gnd = ptcl_utils.get_GndSeg(label_index, GndClasses=[40])
                    pcd[:, 3] = gnd
                    pointclouds.append(pcd)
                    vehicle_ids.append(vehicle_id)
                    pred = ptcl_utils.ransac_predict(pcd, threshold=0.08)
                    pred_grid, _ = ptcl_utils.calculate_grid_label_ransac(1, pred, center=vehicle_positions[vehicle_id])
                    np.savetxt('local-prediction/%s-%d.txt' % (vehicle_id, f_id), pred_grid)

            # merged = np.vstack(pointclouds)
            # print(merged.shape)
            # # get ground truth grid for each vehicle
            # for v_id, pos in vehicle_positions.items():
            #     gt_grid, _ = ptcl_utils.calculate_grid_label_ransac(1, merged, center=pos)
            #     np.savetxt('ground-truth-grid/%s-%d.txt'%(v_id, f_id), gt_grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to visualize point clouds")
    parser.add_argument("--datadir", type=str, default="", help="Path to the dataset folder")
    parser.add_argument("--frame", type=int, default=-1, help="Frame id (must specify datadir)")
    parser.add_argument("--label", type=str, default="", help="Path to the file of labels (optional)")
    parser.add_argument("--pcd", type=str, default=[], nargs="+",
                        help="Path to the files of pointclouds (at least one)")
    # Not implemented
    parser.add_argument("--show", action="store_true", default=True, help="Show the figure on display")
    parser.add_argument("--save", type=str, default="", help="Save the figure to the file specified")
    args = parser.parse_args()

    main(args)
