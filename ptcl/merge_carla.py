import numpy as np
import os
import pcd_merge

carla_dataset_dir = '/home/mininet-wifi/Carla/lidar'
vehicle_ids = os.listdir(carla_dataset_dir)
start_frame_idx = 1000
total_frames = 80

for i in range(start_frame_idx, start_frame_idx + total_frames):
    print(i)
    pcds, oxtses = [], []
    for id in vehicle_ids:
        data_path = os.path.join(carla_dataset_dir, id, '%d.npy'%i)
        oxts_path = os.path.join(carla_dataset_dir, id, '%d.trans.npy'%i)
        pcd = np.load(data_path)
        oxts = np.load(oxts_path)
        pcds.append(pcd)
        oxtses.append(oxts)
    merged_pcd = pcd_merge.merge_carla(pcds, oxtses)
    print(merged_pcd.dtype)
    merged_pcd = merged_pcd.astype(np.float32)
    print(merged_pcd.dtype)
    with open('/home/mininet-wifi/Carla/merged/%d.bin'%i, 'w') as f:
        merged_pcd.tofile(f)
    # np.save('/home/mininet-wifi/Carla/merged/%d.npy'%i, merged_pcd)
