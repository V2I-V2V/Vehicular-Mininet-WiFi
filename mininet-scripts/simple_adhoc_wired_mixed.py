import sys

from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mn_wifi.link import wmediumd, adhoc, Intf
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.wmediumdConnector import interference


def topology(args):
    net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)

    info( '*** Add switches\n')
    s1 = net.addSwitch('s1')

    info("*** Creating nodes\n")
    server = net.addHost('server', mac='00:00:00:00:00:01', ip6='fe80::3', 
                        position='15,15,0')
    sta1 = net.addStation('sta1', mac='00:00:00:00:00:02', ip6='fe80::1',
                        position='10,10,0')
    sta2 = net.addStation('sta2', mac='00:00:00:00:00:03', ip6='fe80::2',
                        position='50,10,0')                        


    net.setPropagationModel(model="logDistance", exp=4)

    # plot the topology
    info("*** Plotting Graph\n")
    net.plotGraph(max_x=200, max_y=200)

    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()

    info("*** Creating links\n")
    # MANET routing protocols supported by proto:
    # babel, batman_adv, batmand and olsr
    # WARNING: we may need to stop Network Manager if you want
    # to work with babel
    protocols = ['babel', 'batman_adv', 'batmand', 'olsrd', 'olsrd2']
    kwargs = dict()
    for proto in args:
        if proto in protocols:
            kwargs['proto'] = proto
    
    net.addLink(sta1, s1, cls=TCLink, intf='sta1-eth1')
    net.addLink(server, s1, cls=TCLink)
    net.addLink(sta1, cls=adhoc, intf='sta1-wlan0',
                ssid='adhocNet', mode='g', channel=5,
                **kwargs)
    net.addLink(sta2, cls=adhoc, intf='sta2-wlan0',
                ssid='adhocNet', mode='g', channel=5,
                **kwargs)
    
    info("\n*** Addressing...\n")
    if 'proto' not in kwargs:
        sta1.setIP6('2001::1/64', intf="sta1-wlan0")
        sta2.setIP6('2001::2/64', intf="sta2-wlan0")
        sta1.setIP('10.0.0.4/8', intf="sta1-eth1")
        # server.setIP6('2001::3/64', intf="server-eth0")
    
    info("*** Starting network\n")
    net.build()

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    topology(sys.argv)