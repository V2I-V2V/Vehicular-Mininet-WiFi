# -*- coding: utf-8 -*-
# Filename : visualization

import numpy as np
import open3d as o3d
import sys
import ptcl_utils
import argparse


def main(args):
	filepath = args.filepath
	ptcl = ptcl_utils.read_ptcl_data(filepath)
	print(ptcl.shape)
	render, save, save_path = True, False, None
	if args.no_render:
		print("no rendering")
		render = False
	if args.savepath != "":
		print("save")
		save = True
		save_path = args.savepath
	ptcl_utils.draw_ptcl(ptcl, args.mode, show=render, save=save, save_path=save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Script to visualize point clouds")
	parser.add_argument("--filepath", type=str, help="Path to the point cloud file")
	parser.add_argument("--mode", type=str, default="3d", choices=["3d", "2d"])
	parser.add_argument("--savepath", type=str, default="", help="path to save")
	parser.add_argument("--no_render", action="store_true", default=False, help="Do not render")
	args = parser.parse_args()
	main(args)


### TODO:

# read, write, visualize, draw bounding boxes

# visualize (2D: matplotlib, 3D: Open3D)

# Put Gnd detection repo into the ptcl/dir



