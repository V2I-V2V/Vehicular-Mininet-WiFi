#!/usr/bin/python

"Example to create a Mininet-WiFi topology and connect it to the internet via NAT"

from mininet.node import Controller
from mininet.log import setLogLevel, info
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi


def topology():

    "Create a network."

    net = Mininet_wifi(controller=Controller)

    info("*** Creating nodes\n")
    ap1 = net.addAccessPoint('ap1', ssid='new-ssid', mode='g', channel='1', position='10,10,0')
    sta1 = net.addStation('sta1', position='10,20,0')
    c1 = net.addController('c1', controller=Controller)

    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()

    info("*** Starting network\n")
    net.build()
    net.addNAT(name='nat0', linkTo='sta1', ip='192.168.100.254').configDefault()
    c1.start()
    ap1.start([c1])

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()

    
if __name__ == '__main__':
    setLogLevel('info')
    topology()
