import os
from numba.cuda.simulator.api import detect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

matplotlib.rc('font', **{'size': 18})


single_vehicle_dir = './gta/single_vehicle/'
merged_vehicle_dir = './gta/merged/'

single_vehicle_frames = os.listdir(single_vehicle_dir)
merged_vehicle_frames = os.listdir(merged_vehicle_dir)

single_area_arr, merged_area_arr = [], []

def calculate_grid_label(grid_size, points):
    x_size, y_size = int(100/grid_size), int(100/grid_size)
    grid = np.zeros((x_size, y_size), dtype=int)
    # for x_id in range(x_size):
    #     for y_id in range(y_size):
    #         in_range_points = points[points[:, 0] >= ]
    for point in points:
        x_idx, y_idx = int((point[0]+50)/grid_size), int((point[1]+50)/grid_size)
        if point[3] == 1:
            # obj
            grid[x_idx][y_idx] -= 1
        elif point[3] == 0:
            # ground 
            grid[x_idx][y_idx] += 1
    
    grid[grid > 0] = 1 # drivable
    grid[grid < 0] = -1 # object
    # print(len(grid[grid > 0]))
    
    return grid


# for frame_idx in range(1, 7):  #len(single_vehicle_frames)

#     # print(frame_idx)
#     single_vehicle_f = np.load("00_%sv"%str(frame_idx)+'.npy') # single_vehicle_dir+
#     # merged_vehicle_f = np.load(str(frame_idx)+'.npy') # merged_vehicle_dir+
#     # merged_vehicle_f = np.load(str(frame_idx)+'.npy')
#     single_vehicle_grid = calculate_grid_label(1, single_vehicle_f)
#     # np.savetxt('%d_grid_label.txt'%frame_idx, single_vehicle_grid)
#     # merged_vehicle_grid = calculate_grid_label(1, merged_vehicle_f)
#     # np.savetxt('%d_grid_label_merged.txt'%frame_idx, merged_vehicle_grid)
#     single_area = len(single_vehicle_grid[single_vehicle_grid != 0]) * 1
#     # merged_area = len(merged_vehicle_grid[merged_vehicle_grid != 0]) * 1
#     single_area_arr.append(single_area)
#     # merged_area_arr.append(merged_area)
#     print(single_area)


# fig = plt.figure(figsize=(6,4))
# ax = fig.add_subplot(111)

# ax.plot(np.arange(len(single_area_arr)), single_area_arr, '--o', label='single vehicle')
# # ax.plot(np.arange(len(single_area_arr)), merged_area_arr, '-o', label='merged')
# ax.set_xticks(np.arange(len(single_area_arr)))
# ax.set_xticklabels(['1V', '2V', '3V', '4V', '5V', '6V'])
# ax.legend()

# plt.xlabel('Number of vehicles')
# plt.ylabel('Space Area Detected ($m^2$)')
# plt.tight_layout()
# plt.savefig('detection_space.png')


result_dir = './gta/decoded_grid_labels/'

detected_area = {}
for i in range(80):
    prefix = '%06d_'%i
    for v_num in range(1, 7):
        result_file = result_dir + prefix + '%d.txt'%v_num
        grids = np.loadtxt(result_file)
        if v_num not in detected_area:
            detected_area[v_num] = [len(grids[grids != 0])]
        else:
            detected_area[v_num].append(len(grids[grids != 0]))

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
for v_num in range(1, 7):
    ax.plot(np.arange(len(detected_area[v_num])), detected_area[v_num], '--.', label='%dV'%v_num)
plt.legend()
plt.xlabel('Frame number')
plt.ylabel('Space Area Detected ($m^2$)')
plt.tight_layout()
plt.savefig('detection_diff_vehicles.png')

result_dir = './gta/grid_labels_all_comb/'
files = os.listdir(result_dir)
detected_area = {}
for f in files:
    prefix = f[:6]
    comb = f[7:-4]
    grids = np.loadtxt(result_dir+f)
    if comb not in detected_area:
        detected_area[comb] = [len(grids[grids != 0])]
    else:
        detected_area[comb].append(len(grids[grids != 0]))

        


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
# for v_num in range(1, 7):
#     ax.plot(np.arange(len(detected_area[v_num])), detected_area[v_num], '--.', label='%dV'%v_num)
summarized_area = {}
best_comb, worst_comb = {}, {}
for comb, area in detected_area.items():
    if len(comb) not in summarized_area:
        summarized_area[len(comb)] = np.array(area)
        best_comb[len(comb)] = (comb, np.mean(area))
        worst_comb[len(comb)] = (comb, np.mean(area))
    else:
        summarized_area[len(comb)] = np.vstack((summarized_area[len(comb)], area))
        if np.mean(area) > best_comb[len(comb)][1]:
            best_comb[len(comb)] = (comb, np.mean(area))
        if np.mean(area) < worst_comb[len(comb)][1]:
            worst_comb[len(comb)] = (comb, np.mean(area))

print(best_comb)
print(worst_comb)

for v_num, area in sorted(summarized_area.items()):
    print(v_num, area.shape)

    if v_num < 6:
        avg = np.mean(area, axis=0)
        std = np.std(area, axis=0)
        print(avg.shape)
        print(avg.dtype)
        ax.scatter(np.arange(avg.shape[0]), avg, label='%dV'%v_num)
        ax.errorbar(np.arange(avg.shape[0]), avg, yerr=std, capsize=3)

    else:
        ax.scatter(np.arange(len(area)), area, label='%dV'%v_num)
        ax.errorbar(np.arange(len(area)), area, yerr=np.zeros((len(area),)))
    # print(comb, len(area))
    # ax.plot(np.arange(len(area)), area, '--.', label=comb)
plt.legend()
plt.xlabel('Frame number')
plt.ylabel('Space Area Detected ($m^2$)')
plt.tight_layout()
plt.savefig('detection_diff_vehicles_all.png')


fig, axes = plt.subplots(6, 1, figsize=(10,10), sharex=True)

for v_num in best_comb:
    best_key, worst_key = best_comb[v_num][0], worst_comb[v_num][0]
    axes[v_num-1].plot(np.arange(80), detected_area[best_key], '--.', label=best_key)
    axes[v_num-1].plot(np.arange(80), detected_area[worst_key], '--.', label=worst_key)
    
    if v_num == 3:
        axes[v_num-1].set_ylabel('Space Area Detected ($m^2$)')
    if v_num == 4:
        axes[v_num-1].plot(np.arange(80), detected_area['13'], '--.', label='13')
        axes[v_num-1].plot(np.arange(80), detected_area['045'], '--.', label='045')
    axes[v_num-1].legend()

plt.xlabel('Frame number')
plt.savefig('detection_combination.png')
