# Harbor: Boosting Collaborative Vehicular Perception on the Edge with Vehicle-to-Vehicle Communication
Scripts to run mininet-wifi experiments on vehicular perception.

## Intsall Mininet-WiFi with python3 (Skip if already installed)

```
git clone https://github.com/intrig-unicamp/mininet-wifi.git ~/mininet-wifi
pip3 install -r requirement.txt
cd ~/mininet-wifi
git checkout 66a20d80063b83111df950762d260774a38d620a
cp ~/Vehicular-Mininet-WiFi/mininet-scripts/install.sh ~/mininet-wifi/util/install.sh # this install.sh has been modified to work with only python3
sudo util/install.sh -Wlnfv
```

### Update Kernel to v5.8

```
bash update_kernel.sh
```
After the script, reboot the machine. Check the output of `uname -r`, it should be `5.8.0-050800-generic`.

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

## Use simple custom routing

### Static
First, compile the `simple-route.c` by
```
cd routing
make
```
And then start a simple adhoc topogy without routing by

```
python3 routing/adhoc.py
```

From mininet CLI, first verify that sta1 cannot ping sta3 (because thery are out of wireless range and require a intermidiate hop sta2 to route):

```
sta1 ping sta3
PING 10.0.0.3 (10.0.0.3) 56(84) bytes of data.
From 10.0.0.1 icmp_seq=1 Destination Host Unreachable
From 10.0.0.1 icmp_seq=2 Destination Host Unreachable
From 10.0.0.1 icmp_seq=3 Destination Host Unreachable
From 10.0.0.1 icmp_seq=4 Destination Host Unreachable
From 10.0.0.1 icmp_seq=5 Destination Host Unreachable
```

Then type `xterm sta1 sta3` to open two xterm terminals. Execute below command in the corresponding xtem terminal.

In sta1 terminal, execute
```
./routing/simple-route add to 10.0.0.3 dev sta1-wlan0 via 10.0.0.2
```
In sta3 terminal, execute
```
./routing/simple-route add to 10.0.0.1 dev sta3-wlan0 via 10.0.0.2
```

Now try `sta1 ping sta3` and it should work. You can also verify the routing change by looking at the output of `route -n` before/after execute the `simple-route` command.

For more information about the usage of `routing/simple-route.c`, refer to `routing/README.md`.


### Dynamic Routing

Add the argument `-r custom` when running `vehicular_perception.py`. See optional arguments in the next section.


## Use the `vehicular_perception.py` script

Only setup the network:

```
sudo python3 mininet-scripts/vehicular_adhoc_setup.py -n <num_nodes> -p <pcd_data_location_file> <optional options>
```

For example, to start a emulated network without running `vehicle.py`, try

```
sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-multihop.txt --trace input/traces/trace-all-nodes.txt
```

To start the emulated network with `vehicle.py` running, try

```
sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-multihop.txt --trace input/traces/trace-all-nodes.txt --run_app -s minDist --collect-traffic
```

To use fixed scheduler, 

```
sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-multihop.txt --trace input/traces/trace-all-nodes.txt --run_app -s fixed ./input/assignments/assignments-sample.txt <index>
```


Required arguments:

* `-n`: Number of nodes (vehicles), must be `<= 6`.

Optional arguments:

* `--run_app`: Start the vehicular application.
* `-l <location_file>`: Read a location file trace. File format [sta1_x sta1_y sta2_x, sta2_y ...]. If not specified, `input/locations/location-example.txt` will be used.
* `--trace <network_trace>`: V2I network trace for each node. File format [node0_bw node1_bw ...]. If not specified, each node will begin with 100Mbps V2I bw and not change.
* `-p <pcd_data_file>`: pcd data file to locate each node's point cloud data. If not specified, default pcd data file configuration file `input/pcds/pcd-data-config.txt` will be used.


Optional arguments:

* `--adaptive_encode <encode_type>`: Wether to use adaptive endoding, (encode_type: 0 for no adaptive encoding, 1 for adaptive encdoing, 2 for adaptive encoding but always uses 4 chunks). Default value is 0 (no adaptive encoding).
* `--adapt_frame_skipping`: Enable adaptive frame skipping when a frame sending takes too long. Default is not enabled.
* `--multi <0/1>`: 1 for enable 1 help many, 0 for disable. Default is enabled. 
* `-r <routing_algorithm>`: Support `olsrd` and `custom`. Other input will make nodes run no routing algorithm.
* `-s <scheduler>`: Scheduler algorithm used by `server/server.py`. Default scheduler scheme is `minDist`. If you want to use fixed assignment mode, see next bullet.
* `-s fixed <assignment_file> <assignment_index>`: Fix assignment mode. Assignment file format: each line is an assignment, and `<assignment_index>` is the index of the assignment to test. For example  `-f input/assignments.txt 1` test the second assignment/second line in file `input/assignments.txt`
* `--power <tx_power>`: Set tx power (dBm) for all the nodes, support [0, 20]. Relationship b/w pwoer and coverage are [here](https://docs.google.com/spreadsheets/d/1pQjaUDc78t3qYAO2gj00q1HPb3gae56G-EdcOuoQAsY/edit#gid=1920952781).
* `--fps <fps>`: Framerate of `vechile.py`,  default value is 1
* `--no_control`: Disable control messages of `vehicle.py`. 
* `-t <time_in_seconds>`: Total emulation time of the script. Default is 100 s.
* `--no_control`: Disable vehicle control messages. Make sure the scheduler scheme (e.g. fixed assignment) will not require node to send control messages, otherwise the behavior will be undefined.
* `--helpee_conf <helpee_conf_file>`: You can specify which nodes are helpees by providing a configuration file. Default helpee_conf file is `input/helpee_conf/helpee-nodes.txt` (0 and 1 are helpee nodes). This will be updated later to just read the bw traces and determine which node is helpee.
* `--collect-traffic`: Enable pcap trace.
* `--save-data`: Save undecoded point cloud. 
* `--plot`: Plot nodes.


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
python3 vehicle/vehicle.py <node_num> <dir_to_pointcloud_data> # the last argument is optional
```
where node_num = station number - 1 (e.g. `python3 vehicle/vehicle.py 1 ../DeepGTAV-data/object-0227-1/alt_perspective/0022786/` at sta2).

Note: for the main vehicle (refered as vehicle 0 here), you only need to run `python3 vehicle/vehicle 0` because the the default point cloud dir is set to vehicle 0. For others, you can find the point cloud dirs at `../DeepGTAV-data/object-0227-1/alt_perspective/` (e.g. `0022786/`, `0037122/`, etc).

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


## Try a simple exmaple
```
sudo python3 sta_ap_mode.py
```

This script setup 2 APs and 2STAs (each one associated with one ap). A wired connection is created between the 2APs.
