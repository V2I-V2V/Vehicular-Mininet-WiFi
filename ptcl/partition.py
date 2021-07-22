import numpy as np
import sys, os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointcloud

PCD_DIR = ['../DeepGTAV-data/object-0227-1/velodyne_2/', \
'../DeepGTAV-data/object-0227-1/alt_perspective/0022786/velodyne_2/',\
'../DeepGTAV-data/object-0227-1/alt_perspective/0037122/velodyne_2/',\
'../DeepGTAV-data/object-0227-1/alt_perspective/0191023/velodyne_2/',\
'../DeepGTAV-data/object-0227-1/alt_perspective/0399881/velodyne_2/',\
'../DeepGTAV-data/object-0227-1/alt_perspective/0735239/velodyne_2/']

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
    for i in range(len(PCD_DIR)):
        pcd = np.fromfile(PCD_DIR[i]+sys.argv[1], dtype=np.float32).reshape([-1, 4])
        t_s = time.time()
        partitioned_pcd = simple_partition(pcd, 20, sample_rate=16)
        partitioned_pcd = pcd
        print("ptcl shape " + str(partitioned_pcd.shape))
        pc, _ = pointcloud.dracoEncode(partitioned_pcd, 10, 12)
        print("Encoded size:", len(pc))
        partitioned_pcd = simple_partition_no_sample(pcd, 20)
        print("sampled shape " + str(partitioned_pcd.shape))
        pc, _ = pointcloud.dracoEncode(partitioned_pcd, 10, 12)    
        print("Encoded size:", len(pc))
        partitioned_pcd, partitioned_pcd_1, partitioned_pcd2, partitioned_pcd3 = layered_partition(pcd, [5, 8, 15])
        print("layered shape " + str(partitioned_pcd.shape))
        t_elapsed = time.time() - t_s
        print(t_elapsed)
        print(pcd.shape)
        print(partitioned_pcd.shape)
        print(partitioned_pcd.shape[0]/pcd.shape[0])
        pcd, _ = pointcloud.dracoEncode(partitioned_pcd, 10, 12)
        print(len(pcd))
        # with open('original.bin', 'w') as f:
        #     pcd.tofile(f)
        # with open('sample%d.bin'%i, 'w') as f:
        #     partitioned_pcd.tofile(f)
