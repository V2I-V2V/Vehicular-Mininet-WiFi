import sys, os

from numpy.lib import utils

from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mn_wifi.link import wmediumd, adhoc, Intf
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.wmediumdConnector import interference
from threading import Thread as thread
import numpy as np
import time
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

default_loc = ['280.225, 891.726, 0', '313.58, 855.46, 0', '286.116, 832.733, 0',\
                '320.134, 854.744, 0', '296.692, 832.28, 0', '290.943, 881.713, 0', \
                '313.943, 891.713, 0', '312.943, 875.713, 0']
default_loc_file="input/object-0227-loc.txt"

def read_v2i_traces(trace_file):
    trace = np.loadtxt(trace_file)
    return trace[:, 1]


def replay_trace(node, ifname, trace):
    intf = node.intf(ifname)
    time.sleep(2)
    for throughput in trace:
        start_t = time.time()
        intf.config(bw=(throughput+0.001))
        elapsed_t = time.time() - start_t
        sleep_t = 1 - elapsed_t
        time.sleep(sleep_t)


def replay_trace_thread_on_sta(sta, ifname, trace_file):
    thrpt_trace = read_v2i_traces(trace_file)
    replay_thread = thread(target=replay_trace, args=(sta, ifname, thrpt_trace))
    replay_thread.daemon = True
    replay_thread.start()


def replay_fixed_assignment(server, stations, assignment):
    server_cmd = "python3 server.py -f " + assignment + "> logs/server.log 2>&1 &"
    server.cmd(server_cmd)
    v1_cmd = 'sleep 6 && python3 vehicle.py 0 > logs/node0.log 2>&1 &'
    v2_cmd = 'sleep 6 && python3 vehicle.py 1 ../DeepGTAV-data/object-0227-1/alt_perspective/0022786/ \
                    > logs/node1.log 2>&1 &'
    v3_cmd = 'sleep 6 && python3 vehicle.py 2 ../DeepGTAV-data/object-0227-1/alt_perspective/0037122/ \
                    > logs/node2.log 2>&1 &'
    v4_cmd = 'sleep 6 && python3 vehicle.py 3 ../DeepGTAV-data/object-0227-1/alt_perspective/0191023/ \
                    > logs/node3.log 2>&1 &'
    v5_cmd = 'sleep 6 && python3 vehicle.py 4 ../DeepGTAV-data/object-0227-1/alt_perspective/0399881/ \
                    > logs/node4.log 2>&1 &'
    v6_cmd = 'sleep 6 && python3 vehicle.py 5 ../DeepGTAV-data/object-0227-1/alt_perspective/0735239/ \
                    > logs/node5.log 2>&1 &'
    stations[0].cmd(v1_cmd)
    stations[1].cmd(v2_cmd)
    stations[2].cmd(v3_cmd)
    stations[3].cmd(v4_cmd)
    stations[4].cmd(v5_cmd)
    stations[5].cmd(v6_cmd)


def topology(args, locations=default_loc, loc_file=default_loc_file, assignment_str=None):
    net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)
    info("*** Creating nodes\n")
    
    server = net.addHost('server', mac='00:00:00:00:00:01', ip='192.168.0.1/24')
    
    stations = []
    sta1 = net.addStation('sta1', mac='00:00:00:00:00:02', ip6='fe80::1',
                    position=locations[0])
    stations.append(sta1)
    sta2 = net.addStation('sta2', mac='00:00:00:00:00:03', ip6='fe80::2',
                    position=locations[1]) 
    stations.append(sta2)
    sta3 = net.addStation('sta3', mac='00:00:00:00:00:04', ip6='fe80::3',
                    position=locations[2])
    stations.append(sta3)
    sta4 = net.addStation('sta4', mac='00:00:00:00:00:05', ip6='fe80::4',
                    position=locations[3])
    stations.append(sta4)
    sta5 = net.addStation('sta5', mac='00:00:00:00:00:06', ip6='fe80::5',
                    position=locations[4])
    stations.append(sta5)
    sta6 = net.addStation('sta6', mac='00:00:00:00:00:07', ip6='fe80::6',
                    position=locations[5])
    stations.append(sta6)
    sta7 = net.addStation('sta7', mac='00:00:00:00:00:08', ip6='fe80::7',
                    position=locations[6])
    stations.append(sta7)
    sta8 = net.addStation('sta8', mac='00:00:00:00:00:09', ip6='fe80::8',
                    position=locations[7])
    stations.append(sta8)
    s1 = net.addSwitch('s1')
    c1 = net.addController('c1')
    
    net.setPropagationModel(model="logDistance", exp=4)


    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()

    info("*** Creating links\n")
    kwargs = dict()
    kwargs['proto'] = 'olsrd'
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

    # plot
    if '-p' in args:
        net.plotGraph(max_x=400, max_y=1100)

    net.addLink(server, s1, cls=TCLink)
    net.addLink(sta1, s1, cls=TCLink, bw=100, delay='10ms')
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

    info("*** Starting network\n")
    net.build()
    c1.start()
    s1.start([c1])

    # trace replaying, use '-t' for replaying traces
    if '-t' in args:
        replay_trace_thread_on_sta(sta1, "sta1-eth1", "input/traces/1.txt")
        replay_trace_thread_on_sta(sta2, "sta2-eth1", "input/traces/2.txt")
        replay_trace_thread_on_sta(sta3, "sta3-eth1", "input/traces/3.txt")
        replay_trace_thread_on_sta(sta4, "sta4-eth1", "input/traces/4.txt")
        replay_trace_thread_on_sta(sta5, "sta5-eth1", "input/traces/5.txt")
        replay_trace_thread_on_sta(sta6, "sta6-eth1", "input/traces/6.txt")
        replay_trace_thread_on_sta(sta7, "sta7-eth1", "input/traces/7.txt")
        replay_trace_thread_on_sta(sta8, "sta8-eth1", "input/traces/8.txt")

    if '--run_app' in args:
        info("\n*** Running vehicuar server\n")
        server_cmd = "python3 server.py > logs/server.log 2>&1 &"
        server.cmd(server_cmd)

        v1_cmd = 'sleep 6 && python3 vehicle.py 0 ../DeepGTAV-data/object-0227-1/ \
                        %s > logs/node0.log 2>&1 &'%loc_file
        v2_cmd = 'sleep 6 && python3 vehicle.py 1 ../DeepGTAV-data/object-0227-1/alt_perspective/0022786/ \
                        %s > logs/node1.log 2>&1 &'%loc_file
        v3_cmd = 'sleep 6 && python3 vehicle.py 2 ../DeepGTAV-data/object-0227-1/alt_perspective/0037122/ \
                        %s > logs/node2.log 2>&1 &'%loc_file
        v4_cmd = 'sleep 6 && python3 vehicle.py 3 ../DeepGTAV-data/object-0227-1/alt_perspective/0191023/ \
                        %s > logs/node3.log 2>&1 &'%loc_file
        v5_cmd = 'sleep 6 && python3 vehicle.py 4 ../DeepGTAV-data/object-0227-1/alt_perspective/0399881/ \
                        %s > logs/node4.log 2>&1 &'%loc_file
        v6_cmd = 'sleep 6 && python3 vehicle.py 5 ../DeepGTAV-data/object-0227-1/alt_perspective/0735239/ \
                        %s > logs/node5.log 2>&1 &'%loc_file
        sta1.cmd(v1_cmd)
        sta2.cmd(v2_cmd)
        sta3.cmd(v3_cmd)
        sta4.cmd(v4_cmd)
        sta5.cmd(v5_cmd)
        sta6.cmd(v6_cmd)
    
    elif '-f' in args:
        replay_fixed_assignment(server, stations, assignment_str)
        

    info("*** Running CLI\n")
    # if '-f' not in args:
    CLI(net)
    # else:
    #     time.sleep(25)

    info("*** Stopping network\n")
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    sta_locs=default_loc
    loc_file=default_loc_file
    assignment_str = None

    if '-f' in sys.argv:
        print("Run in fixed assignment mode")
        assignment_filename = sys.argv[sys.argv.index('-f')+1]
        assignment_idx = int(sys.argv[sys.argv.index('-f')+2])
        all_assignments = np.loadtxt(assignment_filename, dtype=int)
        sel_assignment = all_assignments[assignment_idx]
        assignment_str = utils.produce_assignment_str(sel_assignment)

    if '-l' in sys.argv:
        loc_filename = sys.argv[sys.argv.index('-l')+1]
        loc_file=loc_filename
        locs = np.loadtxt(loc_filename)
        sta_locs = utils.produce_3d_location_arr(locs)
        print(sta_locs)

    topology(sys.argv, locations=sta_locs, loc_file=loc_file, assignment_str=assignment_str)  