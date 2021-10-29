import sys
# import ptcl.ground
from ptcl.pointcloud import dracoEncode, dracoDecode
import ptcl.ptcl_utils
import getpass
import numpy as np

DATASET_DIR = '/home/'+getpass.getuser()+'/Carla/lidar/'

vehicle_id_to_dir = [86, 130, 174, 108, 119, 141, 152, 163, 185, 97]

def calculate_merged_detection_spaces(v_ids, frame_id, qb_dict):
    ptcls, vehicle_pos = [], {}
    for v_id in v_ids:
        ptcl_name = DATASET_DIR + str(vehicle_id_to_dir[int(v_id)%len(vehicle_id_to_dir)]) \
                    + '/' + str(1000+frame_id) 
        pointcloud = ptcl.ptcl_utils.read_ptcl_data(ptcl_name + '.npy')
        trans = np.load(ptcl_name + '.trans.npy')
        encoded, _ = dracoEncode(pointcloud, 10, qb_dict[v_id])
        decoded = dracoDecode(encoded)
        pointcloud = np.concatenate([decoded, np.ones((decoded.shape[0], 1))], axis=1)
        pointcloud = np.dot(trans, pointcloud[:, :4].T).T
        ptcls.append(pointcloud)
        dummy = np.zeros((4, 1))
        dummy[3] = 1
        new_pos = np.dot(trans, dummy).T
        vehicle_pos[int(v_id)]= (new_pos[0,0], new_pos[0,1])
    merged = np.vstack(ptcls)
    merged_pred = ptcl.ptcl_utils.ransac_predict(merged, threshold=0.1)
    detected_spaces = []
    for v_id in v_ids:
        merged_pred_grid, _ = \
            ptcl.ptcl_utils.calculate_grid_label_ransac(1, merged_pred, center=vehicle_pos[int(v_id)])
        detected_spaces.append(len(merged_pred_grid[merged_pred_grid != 0]))

    return detected_spaces

