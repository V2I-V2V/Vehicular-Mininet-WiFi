import numpy as np
import sys
import time

def simple_partition(pcd, range, sample_rate=16):
    xy_square = np.square(pcd[:, :2])
    dist_to_center = np.sum(xy_square, axis=1)
    in_range_mask = dist_to_center <= (range*range)
    out_of_range_mask = dist_to_center > (range*range)
    partitioned_pcd = pcd[in_range_mask]
    out_of_range_points = pcd[out_of_range_mask]
    sampled = out_of_range_points[::sample_rate,:]
    partitioned_pcd = np.vstack((partitioned_pcd, sampled))
    return partitioned_pcd

def simple_partition_no_sample(pcd, range, sample_rate=16):
    xy_square = np.square(pcd[:, :2])
    dist_to_center = np.sum(xy_square, axis=1)
    in_range_mask = dist_to_center <= (range*range)
    out_of_range_mask = dist_to_center > (range*range)
    partitioned_pcd = pcd[in_range_mask]
    out_of_range_points = pcd[out_of_range_mask]
    sampled = out_of_range_points[::sample_rate,:]
    # partitioned_pcd = np.vstack((partitioned_pcd, sampled))
    return partitioned_pcd


def layered_partition(pcd, ranges):
    xy_square = np.square(pcd[:, :2])
    dist_to_center = np.sum(xy_square, axis=1)
    masks = []
    mask1 = dist_to_center <= (ranges[0] * ranges[0])
    mask2 = (ranges[-1] * ranges[-1]) < dist_to_center
    if len(ranges) == 1:
        return pcd[mask1], pcd[mask2]
    masks.append(mask1)
    # print(ranges)
    for cnt, range in enumerate(ranges[1:]):
        # print(cnt, range)
        masks.append(((ranges[cnt] * ranges[cnt]) < dist_to_center) * (dist_to_center <= (ranges[cnt + 1] * ranges[cnt + 1])))
    masks.append(mask2)
    # print(masks)
    return [pcd[mask] for mask in masks]


if __name__ == "__main__":
    PCD_DIR = ['../DeepGTAV-data/object-0227-1/velodyne_2/', \
            '../DeepGTAV-data/object-0227-1/alt_perspective/0022786/velodyne_2/',\
            '../DeepGTAV-data/object-0227-1/alt_perspective/0037122/velodyne_2/',\
            '../DeepGTAV-data/object-0227-1/alt_perspective/0191023/velodyne_2/',\
            '../DeepGTAV-data/object-0227-1/alt_perspective/0399881/velodyne_2/',\
            '../DeepGTAV-data/object-0227-1/alt_perspective/0735239/velodyne_2/']
    for i in range(len(PCD_DIR)):
        pcd = np.fromfile(PCD_DIR[i]+sys.argv[1], dtype=np.float32).reshape([-1, 4])
        t_s = time.time()
        partitioned_pcd = simple_partition(pcd, 20, sample_rate=16)
        with open('original.bin', 'w') as f:
            pcd.tofile(f)
        with open('sample%d.bin'%i, 'w') as f:
            partitioned_pcd.tofile(f)
