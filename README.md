# Vehicular-Mininet-WiFi
Scripts to run mininet-wifi experiments on vehicular perception.

## Intsall Mininet-WiFi with python3 (Skip if already installed)

```
git clone https://github.com/intrig-unicamp/mininet-wifi.git ~/mininet-wifi
cp mininet-scripts/install.sh ~/mininet-wifi/util/install.sh # this install.sh has been modified to work with only python3
cd ~/mininet-wifi
git checkout 66a20d80063b83111df950762d260774a38d620a
sudo util/install.sh -Wlnfv
```

## Important Note

Stop nmetwork-manager before running mininet-wifi scripts (by running the following command).

```
sudo service network-manager stop
```

## Try a simple exmaple
```
sudo python sta_ap_mode.py
```

This script setup 2 APs and 2STAs (each one associated with one ap). A wired connection is created between the 2APs.


## Use the `vehicular_adhoc_setup.py` script

```
sudo python vehicular_adhoc_setup.py
```

Several options:

* Enable mobility: `sudo python vehicular_adhoc_setup.py -m`
* Replay V2I throughput traces: `sudo python vehicular_adhoc_setup.py -t`


## Adding or deleting nodes (on the fly)

### Adding a host 

```
py net.addHost('h3')
py net.addLink(s1, net.get('h3'))
py s1.attach('s1-eth3')
py net.get('h3').cmd('ifconfig h3-eth0 10.3')
```