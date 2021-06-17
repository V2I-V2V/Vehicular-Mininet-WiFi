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

    sta3 = net.addStation('sta3', mac='00:00:00:00:00:04', ip6='fe80::3',
                    position='51,10,0')
    sta4 = net.addStation('sta4', mac='00:00:00:00:00:05', ip6='fe80::4',
                    position='52,10,0')

    sta5 = net.addStation('sta5', mac='00:00:00:00:00:06', ip6='fe80::5',
                    position='53,10,0')
    sta6 = net.addStation('sta6', mac='00:00:00:00:00:07', ip6='fe80::6',
                    position='56,10,0')

    sta7 = net.addStation('sta7', mac='00:00:00:00:00:08', ip6='fe80::7',
                    position='58,10,0')

    sta8 = net.addStation('sta8', mac='00:00:00:00:00:09', ip6='fe80::8',
                    position='59,10,0')
    
    s1 = net.addSwitch('s1')
    c1 = net.addController('c1')
    
    net.setPropagationModel(model="logDistance", exp=4)


    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()

    info("*** Creating links\n")
    kwargs = dict()
    kwargs['proto'] = 'batmand'
    # add adhoc interfaces
    net.addLink(sta1, cls=adhoc, intf='sta1-wlan0',
                ssid='adhocNet', mode='g', channel=5, **kwargs)
    net.addLink(sta2, cls=adhoc, intf='sta2-wlan0',
                ssid='adhocNet', mode='g', channel=5, **kwargs)
    net.addLink(sta3, cls=adhoc, intf='sta3-wlan0',
                ssid='adhocNet', mode='g', channel=5, **kwargs)
    net.addLink(sta4, cls=adhoc, intf='sta4-wlan0',
                ssid='adhocNet', mode='g', channel=5, **kwargs)
    net.addLink(sta5, cls=adhoc, intf='sta5-wlan0',
                ssid='adhocNet', mode='g', channel=5, **kwargs)
    net.addLink(sta6, cls=adhoc, intf='sta6-wlan0',
                ssid='adhocNet', mode='g', channel=5, **kwargs)
    net.addLink(sta7, cls=adhoc, intf='sta7-wlan0',
                ssid='adhocNet', mode='g', channel=5, **kwargs)
    net.addLink(sta8, cls=adhoc, intf='sta8-wlan0',
                ssid='adhocNet', mode='g', channel=5, **kwargs)

    net.addLink(sta1, s1, cls=TCLink)
    net.addLink(server, s1, cls=TCLink)
    net.addLink(sta2, s1, cls=TCLink)
    net.addLink(sta3, s1, cls=TCLink)
    net.addLink(sta4, s1, cls=TCLink)
    net.addLink(sta5, s1, cls=TCLink)
    net.addLink(sta6, s1, cls=TCLink)
    net.addLink(sta7, s1, cls=TCLink)
    net.addLink(sta8, s1, cls=TCLink)


    info("*** Addressing...\n")
    # setup wireless intf
    sta1.setIP6('2001::1/64', intf="sta1-wlan0")
    sta2.setIP6('2001::2/64', intf="sta2-wlan0")
    sta3.setIP6('2001::3/64', intf="sta3-wlan0")
    sta4.setIP6('2001::4/64', intf="sta4-wlan0")
    sta5.setIP6('2001::5/64', intf="sta5-wlan0")
    sta6.setIP6('2001::6/64', intf="sta6-wlan0")
    sta7.setIP6('2001::7/64', intf="sta7-wlan0")
    sta8.setIP6('2001::8/64', intf="sta8-wlan0")
    # serup wired intf
    sta1.setIP('192.168.0.3/24', intf="sta1-eth1")
    sta2.setIP('192.168.0.4/24', intf="sta2-eth1")
    sta3.setIP('192.168.0.5/24', intf="sta3-eth1")
    sta4.setIP('192.168.0.6/24', intf="sta4-eth1")
    sta5.setIP('192.168.0.7/24', intf="sta5-eth1")
    sta6.setIP('192.168.0.8/24', intf="sta6-eth1")
    sta7.setIP('192.168.0.9/24', intf="sta7-eth1")
    sta8.setIP('192.168.0.10/24', intf="sta8-eth1")
    # server.setIP('192.168.0.5/24', intf="server-eth1")

    sta1.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    sta2.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    sta3.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    sta4.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    sta5.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    sta6.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    sta7.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    sta8.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    server.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')

    sta1.cmd('route add -net 10.0.0.0/24 gw 10.0.0.2')
    sta2.cmd('route add -net 10.0.0.0/24 gw 10.0.0.3')
    sta3.cmd('route add -net 10.0.0.0/24 gw 10.0.0.3')
    sta4.cmd('route add -net 10.0.0.0/24 gw 10.0.0.4')
    sta5.cmd('route add -net 10.0.0.0/24 gw 10.0.0.5')
    sta6.cmd('route add -net 10.0.0.0/24 gw 10.0.0.6')
    sta7.cmd('route add -net 10.0.0.0/24 gw 10.0.0.7')
    sta8.cmd('route add -net 10.0.0.0/24 gw 10.0.0.8')


    info("*** Starting network\n")
    net.build()
    c1.start()
    s1.start([c1])

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topology(sys.argv)  