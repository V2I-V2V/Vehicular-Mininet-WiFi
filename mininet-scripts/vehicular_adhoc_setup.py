import sys

from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mn_wifi.link import wmediumd, adhoc, Intf
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.wmediumdConnector import interference
from threading import Thread as thread
import numpy as np
import time

def read_v2i_traces(trace_file):
    trace = np.loadtxt(trace_file)
    return trace[:, 1]

def replay_trace(node, ifname, trace):
    intf = node.intf(ifname)
    time.sleep(20)
    for throughput in trace:
        start_t = time.time()
        intf.config(bw=int(throughput))
        elapsed_t = time.time() - start_t
        sleep_t = 1 - elapsed_t
        time.sleep(sleep_t)

def topology(args):
    net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)
    info("*** Creating nodes\n")
    server = net.addHost('server', mac='00:00:00:00:00:01', ip='192.168.0.1/24',
                        position='288, 881, 0')
    sta1 = net.addStation('sta1', mac='00:00:00:00:00:02', ip6='fe80::1',
                        position='280.225, 891.726, 0')
    sta2 = net.addStation('sta2', mac='00:00:00:00:00:03', ip6='fe80::2',
                        position='313.58, 855.46,0') 

    sta3 = net.addStation('sta3', mac='00:00:00:00:00:04', ip6='fe80::3',
                    position='286.116, 832.733,0')
    sta4 = net.addStation('sta4', mac='00:00:00:00:00:05', ip6='fe80::4',
                    position='320.134, 854.744,0')

    sta5 = net.addStation('sta5', mac='00:00:00:00:00:06', ip6='fe80::5',
                    position='296.692, 832.28, 0')
    sta6 = net.addStation('sta6', mac='00:00:00:00:00:07', ip6='fe80::6',
                    position='290.943, 881.713, 0')

    sta7 = net.addStation('sta7', mac='00:00:00:00:00:08', ip6='fe80::7',
                    position='313.943, 891.713, 0')

    sta8 = net.addStation('sta8', mac='00:00:00:00:00:09', ip6='fe80::8',
                    position='312.943, 875.713, 0')
    
    s1 = net.addSwitch('s1')
    c1 = net.addController('c1')
    
    net.setPropagationModel(model="logDistance", exp=4)


    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()

    info("*** Creating links\n")
    kwargs = dict()
    kwargs['proto'] = 'batmand'
    # add adhoc interfaces
    channel_num = 5
    net.addLink(sta1, cls=adhoc, intf='sta1-wlan0',
                ssid='adhocNet', mode='g', channel=channel_num, **kwargs)
    net.addLink(sta2, cls=adhoc, intf='sta2-wlan0',
                ssid='adhocNet', mode='g', channel=channel_num, **kwargs)
    net.addLink(sta3, cls=adhoc, intf='sta3-wlan0',
                ssid='adhocNet', mode='g', channel=channel_num, **kwargs)
    net.addLink(sta4, cls=adhoc, intf='sta4-wlan0',
                ssid='adhocNet', mode='g', channel=channel_num, **kwargs)
    net.addLink(sta5, cls=adhoc, intf='sta5-wlan0',
                ssid='adhocNet', mode='g', channel=channel_num, **kwargs)
    net.addLink(sta6, cls=adhoc, intf='sta6-wlan0',
                ssid='adhocNet', mode='g', channel=channel_num, **kwargs)
    net.addLink(sta7, cls=adhoc, intf='sta7-wlan0',
                ssid='adhocNet', mode='g', channel=channel_num, **kwargs)
    net.addLink(sta8, cls=adhoc, intf='sta8-wlan0',
                ssid='adhocNet', mode='g', channel=channel_num, **kwargs)
    # net.addNAT(name='nat0', linkTo='s1', ip='192.168.100.1').configDefault()

    # net.addLink(sta1, cls=adhoc, intf='sta1-wlan0',
    #             ssid='adhocNet', mode='g', channel=5, ht_cap='HT40+',  **kwargs)
    # net.addLink(sta2, cls=adhoc, intf='sta2-wlan0',
    #             ssid='adhocNet', mode='g', channel=5, ht_cap='HT40+', **kwargs)
    # net.addLink(sta3, cls=adhoc, intf='sta3-wlan0',
    #             ssid='adhocNet', mode='g', channel=5, ht_cap='HT40+', **kwargs)
    # net.addLink(sta4, cls=adhoc, intf='sta4-wlan0',
    #             ssid='adhocNet', mode='g', channel=5, ht_cap='HT40+', **kwargs)
    # net.addLink(sta5, cls=adhoc, intf='sta5-wlan0',
    #             ssid='adhocNet', mode='g', channel=5, ht_cap='HT40+', **kwargs)
    # net.addLink(sta6, cls=adhoc, intf='sta6-wlan0',
    #             ssid='adhocNet', mode='g', channel=5, ht_cap='HT40+', **kwargs)
    # net.addLink(sta7, cls=adhoc, intf='sta7-wlan0',
    #             ssid='adhocNet', mode='g', channel=5, ht_cap='HT40+', **kwargs)
    # net.addLink(sta8, cls=adhoc, intf='sta8-wlan0',
    #             ssid='adhocNet', mode='g', channel=5, ht_cap='HT40+', **kwargs)

    # plot
    if '-p' not in args:
        net.plotGraph(max_x=400, max_y=1100)

    net.addLink(sta1, s1, cls=TCLink, bw=100, delay='10ms')
    net.addLink(server, s1, cls=TCLink)
    net.addLink(sta2, s1, cls=TCLink, bw=100, delay='10ms')
    net.addLink(sta3, s1, cls=TCLink, bw=100, delay='10ms')
    net.addLink(sta4, s1, cls=TCLink, bw=100, delay='10ms')
    net.addLink(sta5, s1, cls=TCLink, bw=100, delay='10ms')
    net.addLink(sta6, s1, cls=TCLink, bw=100, delay='10ms')
    net.addLink(sta7, s1, cls=TCLink, bw=100, delay='10ms')
    net.addLink(sta8, s1, cls=TCLink, bw=100, delay='10ms')

    # configure mobility
    if '-m' in args:
        net.startMobility(time=0, mob_rep=1, reverse=False)
        p1_start, p2_start, p1_end, p2_end = dict(), dict(), dict(), dict()
        if '-c' not in args:
            p1_start = {'position': '286.116,832.733,0.0'}
            p2_start = {'position': '280.225,891.726,0.0'}
            p1_end = {'position': '257.905,907.762,0.0'}
            p2_end = {'position': '284.816,879.443,0.0'}

        net.mobility(sta1, 'start', time=50, **p1_start)
        net.mobility(sta2, 'start', time=50, **p2_start)
        net.mobility(sta1, 'stop', time=60, **p1_end)
        net.mobility(sta2, 'stop', time=60, **p2_end)
        net.stopMobility(time=61)

    info("*** Addressing...\n")
    # setup wireless intf
    # sta1.setIP6('2001::1/64', intf="sta1-wlan0")
    # sta2.setIP6('2001::2/64', intf="sta2-wlan0")
    # sta3.setIP6('2001::3/64', intf="sta3-wlan0")
    # sta4.setIP6('2001::4/64', intf="sta4-wlan0")
    # sta5.setIP6('2001::5/64', intf="sta5-wlan0")
    # sta6.setIP6('2001::6/64', intf="sta6-wlan0")
    # sta7.setIP6('2001::7/64', intf="sta7-wlan0")
    # sta8.setIP6('2001::8/64', intf="sta8-wlan0")
    # serup wired intf
    sta1.setIP('192.168.0.3/24', intf="sta1-eth1")
    sta2.setIP('192.168.0.4/24', intf="sta2-eth1")
    sta3.setIP('192.168.0.5/24', intf="sta3-eth1")
    sta4.setIP('192.168.0.6/24', intf="sta4-eth1")
    sta5.setIP('192.168.0.7/24', intf="sta5-eth1")
    sta6.setIP('192.168.0.8/24', intf="sta6-eth1")
    sta7.setIP('192.168.0.9/24', intf="sta7-eth1")
    sta8.setIP('192.168.0.10/24', intf="sta8-eth1")

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

    # trace replaying, use '-t' for replaying traces
    if '-t' in args:
        sta1_thrpt_trace = read_v2i_traces("input/lte-trace-example.txt")
        replay_thread = thread(target=replay_trace, args=(sta1, "sta1-eth1", sta1_thrpt_trace))
        replay_thread.daemon = True
        replay_thread.start()

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topology(sys.argv)  