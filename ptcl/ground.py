# -*- coding: utf-8 -*-
# Filename : ground_gta

import numpy as np
import time
import random
import math
import sys
import os
import open3d as o3d

# modified from ground_v3_2. reduce max iterations, consider vehicle height

def my_ransac_v5(data, max_trials, distance_threshold=0.3, P=0.99, sample_size=3, max_iterations=10000, lidar_height=-1.73+0.4, lidar_height_down=-1.73-0.2, alpha_threshold=0.03, use_all_sample=False):

    random.seed(12345)
    max_point_num = -999
    best_model = None
    best_filt = None
    alpha = 999
    i = 0
    K = max_iterations


    if not use_all_sample:
        z_filter = data[:,2] < lidar_height
        z_filter_down = data[:,2] > lidar_height_down
        filt = np.logical_and(z_filter_down, z_filter)

        first_line_filtered = data[filt,:]
    else:
        first_line_filtered = data

    if data.shape[0] < 1900 or first_line_filtered.shape[0] < 180:
        print(' RANSAC point number too small.')
        return np.argwhere(np.ones(len(data))).flatten(), np.argwhere(np.zeros(len(data))).flatten(), None, None

    L_data = data.shape[0]
    R_L = range(first_line_filtered.shape[0])
    # print(data.shape[0], first_line_filtered.shape[0])


    trials = 0
    while i < K:  # i < 10
        if trials > max_trials:
            print(' RANSAC reached the maximum number of trials.')
            return np.argwhere(np.ones(len(data))).flatten(), np.argwhere(np.zeros(len(data))).flatten(), None, None
        trials = trials + 1
        s3 = random.sample(R_L, sample_size)

        coeffs = estimate_plane(first_line_filtered[s3,:], normalize=False)
        if coeffs is None:
            continue
        r = np.sqrt(coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2 )
        alphaz = math.acos(abs(coeffs[2]) / r)

        d = np.divide(np.abs(np.matmul(coeffs[:3], data[:,:3].T) + coeffs[3]), r)
        d_filt = np.array(d < distance_threshold)
        d_filt_object = ~d_filt

        near_point_num = np.sum(d_filt,axis=0)

        if near_point_num > max_point_num and alphaz < alpha_threshold:
            max_point_num = near_point_num

            best_model = coeffs
            best_filt = d_filt
            best_filt_object = d_filt_object

            alpha = alphaz

            w = near_point_num / L_data

            wn = math.pow(w, 3)
            p_no_outliers = 1.0 - wn
            K = (math.log(1-P) / math.log(p_no_outliers))

        # print(i, K, coeffs)
        i += 1


        if i > max_iterations:
            print(' RANSAC reached the maximum number of trials.')
            return None,None,None,None

    # print(i)
    return np.argwhere(best_filt).flatten(),np.argwhere(best_filt_object).flatten(), best_model, alpha

def estimate_plane(xyz, normalize=True):

    vector1 = xyz[1,:] - xyz[0,:]
    vector2 = xyz[2,:] - xyz[0,:]

    if not np.all(vector1[:3]):
        # print('will divide by zero..', vector1)
        return None
    dy1dy2 = vector2 / vector1

    if  not ((dy1dy2[0] != dy1dy2[1])  or  (dy1dy2[2] != dy1dy2[1])):  # three points in line
        return None


    a = (vector1[1]*vector2[2]) - (vector1[2]*vector2[1])
    b = (vector1[2]*vector2[0]) - (vector1[0]*vector2[2])
    c = (vector1[0]*vector2[1]) - (vector1[1]*vector2[0])
    # normalize
    if normalize:
        # r = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        r = math.sqrt(a ** 2 + b ** 2 + c ** 2)
        a = a / r
        b = b / r
        c = c / r
    d = -(a*xyz[0,0] + b*xyz[0,1] + c*xyz[0,2])
    # return a,b,c,d
    return np.array([a,b,c,d])

folds = ['0312', '0313', '0317', '0325', '0326']
data_path = ['/home/mininet-wifi/DeepGTAV-data/object-0227-1/']
ptcl_path = [path+'velodyne_2/' for path in data_path]
self_path = [path+'ego_object/' for path in data_path]
alt_path = [path+'alt_perspective/' for path in data_path]

num_frame = [166, 334, 252, 250, 464]

if __name__ == '__main__':
    max_trials = int(sys.argv[1])

    # data_path = '/home/harry/5g/edge/DeepGTAV-data/object-0325/velodyne_2/'
    # ego_path = '/home/harry/5g/edge/DeepGTAV-data/object-0325/ego_object/'
    # # data_path = '/home/xumiao/KITTI/object/training/velodyne_backup/'
    # # data_path = '/home/harry/5g/edge/KITTI/2011_09_26/2011_09_26_drive_0035/velodyne_points/data/'
    # # save_path = '/home/xumiao/KITTI/object/training/velodyne/'
    # # save_path = '/home/harry/5g/edge/'

    pcl = np.fromfile(sys.argv[2], dtype=np.float32, count=-1).reshape([-1,4])
    print(pcl.shape)
    t1 = time.time()
    p2, p1, best_model, _ = my_ransac_v5(pcl, max_trials, P=0.8, distance_threshold=0.1, lidar_height=-2.03727+0.1, lidar_height_down=-2.03727-0.1, use_all_sample=True)  # pcl, P=0.8, distance_threshold=0.15, lidar_height=-h+0.05 0.15, lidar_height_down=-h-0.15 -0.2
    print(best_model)
    t2 = time.time()
    print(t2-t1)
    pcl1 = pcl[p1]  # object
    pcl2 = pcl[p2]  # ground
    ratio = len(pcl1)/len(pcl)
    print(ratio)

    # find points below the plane
    print(best_model)
    val = pcl[:, :3].dot(best_model[:3]) + best_model[3]
    less = val <= 0
    below_points = np.argwhere(less).flatten()
    print(below_points.shape)
    pcl_below = pcl[below_points, :]
    print(pcl_below.shape)

    # label the detection pcds
    pcl[:, 3] = 0
    pcl[p2, 3] = 1
    np.save(sys.argv[1], pcl)
    # pcl[below_points, 3] = 0





    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])
    pcd.paint_uniform_color([1, 0, 0])
    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(pcl2[:,:3])
    pcd_ground.paint_uniform_color([0, 0, 1])
    pcd_below = o3d.geometry.PointCloud()
    pcd_below.points = o3d.utility.Vector3dVector(pcl_below[:,:3])
    pcd_below.paint_uniform_color([1, 0, 1])
    o3d.visualization.draw_geometries([pcd, pcd_ground])


    # times = []
    # ratios = []
    # for foldIdx in range(5):
    #     print(folds[foldIdx])
    #     for fID in range(num_frame[foldIdx]):
    #         # print(folds[foldIdx], fID)
    #         vehs = [[0,0]]
    #         frameID = f'{fID:06d}'

    #         pcl = np.fromfile(ptcl_path[foldIdx]+frameID+'.bin', dtype=np.float32, count=-1).reshape([-1,4])
    #         with open(self_path[foldIdx]+frameID+'.txt', 'r') as f:
    #             h = float(f.read().split(' ')[8])
    #         t1 = time.time()
    #         p2, p1, _, _ = my_ransac_v5(pcl, max_trials, P=0.8, distance_threshold=0.15, lidar_height=-h+0.05, lidar_height_down=-h-0.15)  # pcl, P=0.8, distance_threshold=0.15, lidar_height=-h+0.05 0.15, lidar_height_down=-h-0.15 -0.2
    #         t2 = time.time()
    #         times.append(t2-t1)
    #         pcl1 = pcl[p1]  # object
    #         pcl2 = pcl[p2]  # ground
    #         ratio = len(pcl1)/len(pcl)
    #         ratios.append(ratio)

    #         for folder in os.listdir(alt_path[foldIdx]):
    #             if os.path.exists(alt_path[foldIdx]+folder+'/velodyne_2/'+frameID+'.bin'):
    #                 pcl = np.memmap(alt_path[foldIdx]+folder+'/velodyne_2/'+frameID+'.bin', dtype='float32', mode='r').reshape([-1,4])
    #                 with open(alt_path[foldIdx]+folder+'/ego_object/'+frameID+'.txt', 'r') as f:
    #                     h = float(f.read().split()[8])
    #                 t1 = time.time()
    #                 p2, p1, _, _ = my_ransac_v5(pcl, max_trials, P=0.8, distance_threshold=0.15, lidar_height=-h+0.05, lidar_height_down=-h-0.15)  # pcl, P=0.8, distance_threshold=0.15, lidar_height=-h+0.05 0.15, lidar_height_down=-h-0.15 -0.2
    #                 t2 = time.time()
    #                 times.append(t2-t1)
    #                 pcl1 = pcl[p1]  # object
    #                 pcl2 = pcl[p2]  # ground
    #                 ratio = len(pcl1)/len(pcl)
    #                 ratios.append(ratio)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])
            # pcd.paint_uniform_color([0, 0, 1])
            # pcd_object = o3d.geometry.PointCloud()
            # pcd_object.points = o3d.utility.Vector3dVector(pcl1[:,:3])
            # pcd_object.paint_uniform_color([1, 0, 0])
            # o3d.visualization.draw_geometries([pcd, pcd_object])

    print('distance_threshold=0.1')
    # print('Time: AVERAGE {:.3f}, STDEV {:.3f}, MAX {:.3f}, MIN {:.3f}'.format(1000*np.mean(times), 1000*np.std(times), 1000*max(times), 1000*min(times)))
    # print('Ratio: AVERAGE {:.3f}, STDEV {:.3f}, MAX {:.3f}, MIN {:.3f}'.format(100*np.mean(ratios), 100*np.std(ratios), 100*max(ratios), 100*min(ratios)))