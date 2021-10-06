# Point cloud process/visualization scripts

## Usage

### `viz.py`

Assume executing the script under `ptcl/` directory

```bash
python3 viz.py --filepath <ptcl file to show>
```

Required arguments:

* `--filepath <file>`: ptcl data file to plot/save.

Optional arguments:

* `--mode <3d/2d>`: plot in 3D mode or 2D mode.
* `--savepath <path>`: path to save point cloud plot
* `--no_render`: disable plot rendering


### `pcd_process.py`
Assume the `Vehicular-Mininet-WiFi` is put at home directory.
```
python3 ~/Vehicular-Mininet-WiFi/ptcl/pcd_process.py <process_type> <data_folder>
```
Example:
```
python3 ~/Vehicular-Mininet-WiFi/ptcl/pcd_process.py ref ~/gta_ref_frames/
python3 ~/Vehicular-Mininet-WiFi/ptcl/pcd_process.py dis ~/v2x/data-08011700/
```

Required arguments:

* `<process_type>`: reference (ref) or distortion (dis) point clouds.
* `<data_folder>`: the directory to store processed point cloud data. For "dis" process type, it should be the experiment data folder containin distortion point clouds.


For "ref" process_type, This will generate a `data_folder` directory with point clouds processed from GTA5 data (currently do not support Carla data).