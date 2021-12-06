iface="$1" # wireless interface to config
chan="$2" # channel number
ip_addr="$3" # ip addr assigned (last 8bits)
# Preparation
sudo service network-manager stop
sudo ip link set ${iface} down
# Configuration
sudo iwconfig ${iface} mode ad-hoc
sudo iwconfig ${iface} channel ${chan}
sudo iwconfig ${iface} essid 'adhoc-v2v'
sudo iwconfig ${iface} key 1234567890
# activation
sudo ip link set ${iface} up
sudo ip addr add 10.42.0.${ip_addr}/24 dev ${iface}

