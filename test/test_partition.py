import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ptcl.partition import *
from ptcl.pointcloud import *


if __name__ == "__main__":
    pcd = np.fromfile(PCD_DIR[0] + "000000.bin", dtype=np.float32).reshape([-1, 4])
    partitioned_pcd = simple_partition(pcd, 20)
    # partitions = layered_partition(pcd, [5, 8, 15])
    partitions = layered_partition(pcd, [5, 8, 15])
    sum_encoded = b''
    for partition in partitions:
        encoded, ratio = dracoEncode(partition, 10, 8)
        sum_encoded += encoded
        print(len(sum_encoded), len(encoded), len(partition), ratio)
    exit(0)
    for i in range(2, 6):
        partitions = layered_partition(pcd, [i])
        encoded, ratio = dracoEncode(partitions[0], 10, 12)
        print(i, len(encoded), len(partitions[0]), ratio)
    encoded, ratio = dracoEncode(pcd, 10, 12)
    print(i, len(encoded), len(pcd), ratio)

    # for partition in partitions:
    #     print(partition.shape)
    # print(pcd.shape)
    # sum = 0
    # for i in range(4):
        # with open('partition_' + str(i) + ".bin", 'w') as f:
        #     partitioned_pcd[i].tofile(f)
    #     print(partitions[i].shape)
    #     sum += partitions[i].shape[0]
    # print(sum)
