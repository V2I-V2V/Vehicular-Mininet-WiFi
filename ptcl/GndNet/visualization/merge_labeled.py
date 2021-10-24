import numpy as np

label86 = np.load('000000-ground-truth-label.npy')
label130 = np.load('1000-130-ground-truth-label.npy')
non_ground_index = np.argwhere(label130[:, 3] != 1).reshape((-1))
print(non_ground_index.shape, label130.shape[0])
label130[:, 3] = 1

trans86 = np.load('/home/ryanzhu/V2I+V2V/carla-bin/lidar/86/1000.trans.npy')
trans130 = np.load('/home/ryanzhu/V2I+V2V/carla-bin/lidar/130/1000.trans.npy')
# index = np.isin(label130[:, 3], [1])
# print(index.shape)

world_axis = np.dot(label130, trans130.T)
ref_axis = np.dot(world_axis, np.linalg.inv(trans86.T))
ref_axis[non_ground_index, 3] = 0.1


ref_axis = np.vstack([label86, ref_axis])
merged_label = np.zeros((ref_axis.shape[0]), dtype=np.uint32)
ground_idx = np.argwhere(ref_axis[:, 3] == 1).reshape((-1))
merged_label[ground_idx] = 40
merged_label.tofile('86_130_merged.label')

np.save('transform.npy', ref_axis)
with open('86_130_merged.bin', 'w') as f:
    ref_axis = ref_axis.astype(np.float32)
    ref_axis.tofile(f)