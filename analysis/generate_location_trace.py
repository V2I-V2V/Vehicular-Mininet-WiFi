import numpy as np
import pandas as pd
from random import randint
import sys, os

def covert_longitude_latitude_to_x_y(longitude, latitude):
    x, y, zone, ut = utm.from_latlon(latitude, longitude)
    return x, y

def get_random_vehicle_traj(df, vehicle_number_pool):
    randidx = randint(0, len(vehicle_number_pool)-1)
    vehicle_id = vehicle_number_pool[randidx]
    v_traj = df[df['Vehicle_ID'] == vehicle_id].copy(deep=True)
    v_traj = v_traj.sort_values(by=['Global_Time'])
    x, y = v_traj['Global_X'].to_numpy(), v_traj['Global_Y'].to_numpy()
    print(x.shape, y.shape)
    return x, y


df = pd.read_csv('../NGSIM-data.csv', low_memory=False)
print("Number of vehicles", len(df['Vehicle_ID'].unique()))
vehicle_number_pool = df['Vehicle_ID'].to_numpy()

print(df.columns)

print('data shape: ', df.shape)

merged_loc = None
for i in range(6): # make node a argument
    # right now the syncing between 
    x,y = get_random_vehicle_traj(df, vehicle_number_pool)
    if merged_loc is None:
        merged_loc = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
    else:
        min_size = min(merged_loc.shape[0], x.shape[0])
        merged_loc = np.hstack((merged_loc[:min_size], x.reshape(-1,1)[:min_size]))
        merged_loc = np.hstack((merged_loc[:min_size], y.reshape(-1,1)[:min_size]))

print(merged_loc.shape)

# scale to x, y value to 0 by subtracting the minimum
minx = np.min(merged_loc[:, ::2])
# print(minx)
miny = np.min(merged_loc[:, 1::2])
# print(miny)

merged_loc[:, ::2] -= minx
merged_loc[:, 1::2] -= miny

if len(sys.argv) > 1:
    np.savetxt(sys.argv[1]+'.txt', merged_loc, fmt='%f')
else:
    np.savetxt('sample-loc-trace.txt', merged_loc, fmt='%f')



