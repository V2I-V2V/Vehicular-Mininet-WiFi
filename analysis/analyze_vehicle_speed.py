import numpy as np
import sys

if __name__ == '__main__':
    min_speed, max_speed = 0, 0
    trajs = {}
    traj = np.loadtxt(sys.argv[1])
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)

    for i in range(int(traj.shape[1]/2)):
        trajs[i] = traj[:,2*i:2*i+2]
        
    for j in range(1, traj.shape[0]):
        for i in range(int(traj.shape[1]/2)):
            x_end, y_end = traj[:,2*i][j], traj[:,2*i+1][j]
            x_start, y_start = traj[:,2*i][j-1], traj[:,2*i+1][j-1]
            distance_traveled = np.sqrt((y_end-y_start) * (y_end-y_start) + (x_end-x_start) * (x_end-x_start))
            speed = distance_traveled/(0.1) * 2.2369
            if speed > max_speed:
                max_speed = speed
    
    print(max_speed)
            
            