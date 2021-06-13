# Vehicular-Mininet-WiFi
Scripts to run mininet-wifi experiments on vehicular perception.

## Intsall Mininet-WiFi with python3 (Skip if already installed)

```
cd mininet-wifi
sudo util/install.sh -Wlnfv # this install.sh has been modified to work with only python3
```

## Try a simple exmaple
```
sudo python sta_ap_mode.py
```

This script setup 2 APs and 2STAs (each one associated with one ap). A wired connection is created between the 2APs.

