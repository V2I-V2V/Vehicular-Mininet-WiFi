import numpy as np
import yaml
import os

DATASET_DIR = '../data'


result_loc = []

vehicle_ids = []
for vehicle_dir in os.listdir(DATASET_DIR):
    if ".csv" not in vehicle_dir:
        vehicle_ids.append(vehicle_dir)



start_frame_id = 255
end_frame_id = 352


for i in range(255, 352):
    location_line = []
    for v_id in sorted(vehicle_ids):
        with open(os.path.join(DATASET_DIR, v_id, '%d.yaml'%i), 'r') as f:
            pose = yaml.safe_load(f)
        
        location_line.append(pose['lidar_pose'][0])
        location_line.append(pose['lidar_pose'][1])

    result_loc.append(location_line)

locations = np.array(result_loc)
print(locations.shape)

# np.savetxt("location.txt", np.array(result_loc), fmt="%1.3f")


base_str  = "~/data/"
result_data_config = []
for v_id in sorted(vehicle_ids):
    result_data_config.append(base_str + v_id)

np.savetxt("carla-town03-100.txt", np.array(result_data_config), fmt='%s')