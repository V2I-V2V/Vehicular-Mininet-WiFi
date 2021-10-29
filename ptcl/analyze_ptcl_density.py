import numpy as np
import ptcl_utils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', **{'size': 18})

ptcl = ptcl_utils.read_ptcl_data('/home/ryanzhu/MultiVehiclePerception/86_130_merged_10_8.bin')
ptcl_s1 = ptcl_utils.read_ptcl_data('/home/ryanzhu/V2I+V2V/Carla/lidar/86/1000.npy')
ptcl_s2 = ptcl_utils.read_ptcl_data('/home/ryanzhu/V2I+V2V/Carla/lidar/174/1000.npy')

locs = [(-21.642448, 193.882751), (-9.493179, 114.654312), (-5.69124699, 183.77157593)]

ranges = [5, 10, 15, 30, 50, 75, 100]

ptcl_s1 = ptcl_utils.get_points_within_center(ptcl_s1, space_range=100)
ptcl_s2 = ptcl_utils.get_points_within_center(ptcl_s2)
density_s1 = ptcl_utils.avg_point_density_in_range(ptcl_s1, ranges, space_range=100)
density_s2 = ptcl_utils.avg_point_density_in_range(ptcl_s2, ranges, space_range=100)

filter_merged_ptcls1 = ptcl_utils.get_points_within_center(ptcl, space_range=100, center=locs[0])
filter_merged_ptcls2 = ptcl_utils.get_points_within_center(ptcl, space_range=100, center=locs[2])

density_s1_merged = ptcl_utils.avg_point_density_in_range(filter_merged_ptcls1, ranges, space_range=100)
density_s2_merged = ptcl_utils.avg_point_density_in_range(filter_merged_ptcls2, ranges, space_range=100)

print(density_s1)
print(density_s1_merged)
print(density_s2)
print(density_s2_merged)


plt.plot(ranges[2:], density_s1[2:-1], label='single vehicle')
plt.plot(ranges[2:], density_s1_merged[2:-1], label='merged')
plt.xticks(ranges[2:], ['<15', '15-30', '30-50', '50-75', '75-100'])


plt.ylabel('Points per square meter')
plt.xlabel('range (m)')
plt.legend()
plt.show()
