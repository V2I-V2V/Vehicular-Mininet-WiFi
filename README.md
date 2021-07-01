# Vehicular-Mininet-WiFi
Scripts to run mininet-wifi experiments on vehicular perception.

## Intsall Mininet-WiFi with python3 (Skip if already installed)

```
git clone https://github.com/intrig-unicamp/mininet-wifi.git ~/mininet-wifi
pip3 install -r requirements.txt
cp mininet-scripts/install.sh ~/mininet-wifi/util/install.sh # this install.sh has been modified to work with only python3
cd ~/mininet-wifi
git checkout 66a20d80063b83111df950762d260774a38d620a
sudo util/install.sh -Wlnfv
```

## Install routing dependencies

Install BATMAN (-B) routing and OLSR (-O) routing protocol.

```
cd ~/mininet-wifi
sudo util/install.sh -B
sudo util/install.sh -O
```

**Important:** After installing olsr, please open `/etc/olsrd/olsrd.conf` and comment out the line `LoadPlugin "/usr/lib/olsrd_jsoninfo.so.1.1"` if it exists.

As a quick fix, you can copy my config to your olsrd configuration directory by

```
sudo cp config/olsrd.conf /etc/olsrd/
```


## Try a simple exmaple
```
sudo python3 sta_ap_mode.py
```

This script setup 2 APs and 2STAs (each one associated with one ap). A wired connection is created between the 2APs.


## Use the `vehicular_adhoc_setup.py` script

Only setup the network:

```
sudo python3 mininet-scripts/vehicular_adhoc_setup.py -n <num_nodes> -p <pcd_data_location_file> <optional options>
```

For example, to start a emulated network without running `vehicle.py`, try

```
sudo python3 mininet-scripts/vehicular_adhoc_setup.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-multihop.txt --trace input/traces/trace-all-nodes.txt
```

To start the emulated network with `vehicle.py` running, try

```
sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-multihop.txt --trace input/traces/trace-all-nodes.txt --run_app -s minDist --collect-traffic
```

To use fixed scheduler, 

```
sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-multihop.txt --trace input/traces/trace-all-nodes.txt --run_app -s fixed ./input/assignments/assignments-sample.txt <index>
```


Several options:

* Enable pcap trace: `--collect-traffic`
* Plot nodes: `--plot`
* Start the vehicular application: `--run_app`
* Read point cloud data from custom directory: `-p <pcd_config_file>`
* Read a location file: `-l <location_file>`. File format [sta1_x sta1_y sta2_x, sta2_y ...]
* Fix assignment: `-s fixed <assignment_file> <assignment_index>`. Assignment file format: each line is an assignment, and `<assignment_index>` is the index of the assignment to test. For example  `-f input/assignments.txt 1` test the second assignment/second line in file `input/assignments.txt`

## Run experiments over the emulated network

Prepare dataset. Download the dataset from [google drive](https://drive.google.com/file/d/10gjaHto7ZVGs4A2EEVmoLhfUxTDAF3Kw/view?usp=sharing) and unzip it to the parent directory. 

```
unzip DeepGTAV-data.zip && mv DeepGTAV-data ..
```

### Start the application automatically with the script

```
sudo python3 mininet-scripts/vehicular_adhoc_setup.py -n 6 -p input/pcd-data-config.txt -l input/location-multihop.txt --trace input/traces/trace-all-nodes.txt --run_app
```

### Start the application manually

After starting the `python3 mininet-scripts/vehicular_adhoc_setup.py` script, run the follwing commands in the mininet CLI.

```
xterm server sta1 sta2 sta3 sta4 sta5 sta6
```

Then at the server terminal, run
```
python3 server.py 
```

In each station terminal, run 

```
python3 vehicle.py <node_num> <dir_to_pointcloud_data> # the last argument is optional
```
where node_num = station number - 1 (e.g. `python3 vehicle 1 ../DeepGTAV-data/object-0227-1/alt_perspective/0022786/` at sta2).

Note: for the main vehicle (refered as vehicle 0 here), you only need to run `python3 vehicle 0` because the the default point cloud dir is set to vehicle 0. For others, you can find the point cloud dirs at `../DeepGTAV-data/object-0227-1/alt_perspective/` (e.g. `0022786/`, `0037122/`, etc).

## Useful utilities/Troubleshooting

- `sudo mn -c` cleans up mininet setups, useful when something wrong with the network script.

- If you find station pings are failing, you can try to stop nmetwork-manager before running mininet-wifi scripts (by running the following command).

```
sudo service network-manager stop
```

## Adding or deleting nodes (on the fly)

### Adding a host 

```
py net.addHost('h3')
py net.addLink(s1, net.get('h3'))
py s1.attach('s1-eth3')
py net.get('h3').cmd('ifconfig h3-eth0 10.3')
```


### Add hosts in adhoc mode

```
py net.addStation('sta4', ip6='fe80::3', position='120,10,0')
py sta4.setAdhocMode(intf='sta4-wlan0')
```