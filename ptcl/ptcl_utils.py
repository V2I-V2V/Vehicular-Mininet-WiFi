import numpy as np
import open3d as o3d
import sys
import matplotlib.pyplot as plt
import time

def read_ptcl_data(ptcl_fname):
	""" Read point cloud data (.bin/.pcd/.npy) to numpy array 

	Args:
		ptcl_fname ([str]): ptcl file name to process
	"""
	if '.bin' in ptcl_fname:
		ptcl = np.fromfile(ptcl_fname, dtype=np.float32, count=-1).reshape([-1,4])
	elif '.pcd' in ptcl_fname:
		ptcl = o3d.io.read_point_cloud(ptcl_fname)
		ptcl = np.asarray(ptcl.points)
	elif '.npy' in ptcl_fname:
		ptcl = np.load(ptcl_fname)
	return ptcl


def draw_ptcl(pointcloud, mode='3d', show=True, save=False, save_path=None):
	if mode == '3d':
		draw_3d(pointcloud, show, save, save_path)
	elif mode == '2d':
		draw_2d(pointcloud, show, save, save_path)


def draw_3d(pointcloud, show=True, save=False, save_path=None):
	"""Draw point cloud in 3D dimention using open3d

	Args:
		pointcloud ([numpy array]): ptcl data in numpy format
		show (bool, optional): whether to show the figure. Defaults to True.
		save (bool, optional): whether to save the figure. Defaults to False.
		save_path ([type], optional): figure save path when save is True. Defaults to None.
	"""
    # TODO: paint different pcd in various colors? Currently the merge function will produce
	# point clouds from all vehicles
	pointcloud_all = o3d.geometry.PointCloud()
	pointcloud_all.points = o3d.utility.Vector3dVector(pointcloud[:,:3])
	pointcloud_all.paint_uniform_color([0, 0, 1])
	if show:
		o3d.visualization.draw_geometries([pointcloud_all])
	if save:
		vis = o3d.visualization.Visualizer()
		vis.create_window()
		vis.add_geometry(pointcloud_all)
		vis.update_geometry(pointcloud_all)
		vis.poll_events()
		vis.update_renderer()
		time.sleep(1)
		vis.capture_screen_image(save_path, True)
		vis.destroy_window()


def draw_2d(data, show=True, save=False, save_path=None):
	"""Draw point cloud in 3D dimention using matplotlib

	Args:
		data ([np array]): ptcl data in numpy format
		show (bool, optional): whether to show the figure. Defaults to True.
		save (bool, optional): whether to save the figure. Defaults to False.
		save_path ([type], optional): figure save path when save is True. Defaults to None.
	"""
	plt.scatter(data[:,0], data[:,1], s=0.5)
	plt.tight_layout()
	if show:
		plt.show()
	if save:
		plt.savefig(save_path)


def get_number_of_points_in_region(ptcl, x_center, y_center, x_width, y_width):
	""" Get # of points in a region (grid)

	Args:
		ptcl ([np.array]): ptcl data in numpy arr
		x_center ([float]): x center of grid
		y_center ([float]): y center of grid
		x_width ([float]): width of grid in x 
		y_width ([float]): width of grid in y

	Returns:
		[int]: number of points inside grid
	"""
	filtered = ptcl[ptcl[:, 0] > x_center - x_width/2]
	filtered = filtered[filtered[:, 0] < x_center + x_width/2]
	filtered = filtered[filtered[:, 1] > y_center - y_width/2]
	filtered = filtered[filtered[:, 1] < y_center + y_width/2]
	return filtered.shape[0]


def save_ptcl(ptcl, save_path, format='bin'):
	""" Save ptcl (numpy array) to desired format

	Args:
		ptcl ([numpy array]): point cloud data in numpy format
		save_path ([str]): path to save ptcl
		format (str, optional): format to save. Defaults to 'bin'.
	"""
	if format == 'bin':
		with open(save_path, 'w') as f:
			ptcl.tofile(f)
	elif format == 'pcd':
		pointcloud = o3d.geometry.PointCloud()
		pointcloud.points = o3d.utility.Vector3dVector(ptcl[:,:3])
		o3d.io.write_point_cloud(save_path, pointcloud)
	elif format == 'npy':
		np.save(save_path, ptcl)


## TODO: Transform the ptcl to a global reference?