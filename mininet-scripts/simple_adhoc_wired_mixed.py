import sys

from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mn_wifi.link import wmediumd, adhoc, Intf
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.wmediumdConnector import interference


def topology(args):
    net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)
    info("*** Creating nodes\n")
    server = net.addHost('server', mac='00:00:00:00:00:01', ip='192.168.0.1/24',
                        position='15,15,0')
    sta1 = net.addStation('sta1', mac='00:00:00:00:00:02', ip6='fe80::1',
                        position='10,10,0')
    sta2 = net.addStation('sta2', mac='00:00:00:00:00:03', ip6='fe80::2',
                        position='50,10,0') 
    net.setPropagationModel(model="logDistance", exp=4)

    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()

    info("*** Creating links\n")
    kwargs = dict()
    kwargs['proto'] = 'olsrd'
    net.addLink(sta1, cls=adhoc, intf='sta1-wlan0',
                ssid='adhocNet', mode='g', channel=5, **kwargs)
    net.addLink(sta2, cls=adhoc, intf='sta2-wlan0',
                ssid='adhocNet', mode='g', channel=5, **kwargs)
    net.addLink(sta1, server, cls=TCLink)


    info("*** Addressing...\n")
    sta1.setIP6('2001::1/64', intf="sta1-wlan0")
    sta2.setIP6('2001::2/64', intf="sta2-wlan0")
    sta1.setIP('192.168.0.4/24', intf="sta1-eth1")

    sta1.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    server.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')


    info("*** Starting network\n")
    net.build()

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topology(sys.argv)    