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

NUM_NODES = 8
PATICIPATE_NODES = 6 

# point cloud data dir for different partipate vehicle
vehicle_data_dir = ['../DeepGTAV-data/object-0227-1/', 
                    '../DeepGTAV-data/object-0227-1/alt_perspective/0022786/',
                    '../DeepGTAV-data/object-0227-1/alt_perspective/0037122/',
                    '../DeepGTAV-data/object-0227-1/alt_perspective/0191023/',
                    '../DeepGTAV-data/object-0227-1/alt_perspective/0399881/',
                    '../DeepGTAV-data/object-0227-1/alt_perspective/0735239/']
# default start location for different vehicle
default_loc = ['280.225, 891.726, 0', '313.58, 855.46, 0', '286.116, 832.733, 0',\
                '320.134, 854.744, 0', '296.692, 832.28, 0', '290.943, 881.713, 0', \
                '313.943, 891.713, 0', '312.943, 875.713, 0']
default_loc_file="input/object-0227-loc.txt"
default_v2i_bw = [100 for _ in range(NUM_NODES)]
v2i_bw_traces = {}

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
        sleep_t = 1.0 - elapsed_t
        time.sleep(sleep_t)


def replay_trace_thread_on_sta(sta, ifname, thrpt_trace):
    # thrpt_trace = read_v2i_traces(trace_file)
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
    tcpdump_cmd1 = 'tcpdump -nni any -s96 -w node1.pcap &'
    tcpdump_cmd2 = 'tcpdump -nni any -s96 -w node2.pcap &'
    tcpdump_cmd3 = 'tcpdump -nni any -s96 -w node3.pcap &'
    tcpdump_cmd4 = 'tcpdump -nni any -s96 -w node4.pcap &'
    tcpdump_cmd5 = 'tcpdump -nni any -s96 -w node5.pcap &'
    tcpdump_cmd6 = 'tcpdump -nni any -s96 -w node6.pcap &'
    stations[0].cmd(v1_cmd)
    stations[1].cmd(v2_cmd)
    stations[2].cmd(v3_cmd)
    stations[3].cmd(v4_cmd)
    stations[4].cmd(v5_cmd)
    stations[5].cmd(v6_cmd)
    stations[0].cmd(tcpdump_cmd1)
    stations[1].cmd(tcpdump_cmd2)
    stations[2].cmd(tcpdump_cmd3)
    stations[3].cmd(tcpdump_cmd4)
    stations[4].cmd(tcpdump_cmd5)
    stations[5].cmd(tcpdump_cmd6)


def topology(args, locations=default_loc, loc_file=default_loc_file, \
                assignment_str=None, v2i_bw=default_v2i_bw):
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
    for sta_idx in range(len(stations)):
        net.addLink(stations[sta_idx], cls=adhoc, intf='sta%d-wlan0'%(sta_idx+1),
                    ssid='adhocNet', mode='g', channel=channel_num, **kwargs)
        # net.addLink(sta1, cls=adhoc, intf='sta1-wlan0',
        #             ssid='adhocNet', mode='g', channel=channel_num, **kwargs)
    # net.addNAT(name='nat0', linkTo='s1', ip='192.168.100.1').configDefault()

    # add wired interfaces
    net.addLink(server, s1, cls=TCLink)
    for sta_idx in range(len(stations)):
        net.addLink(stations[sta_idx], s1, cls=TCLink, bw=v2i_bw[sta_idx], delay='10ms')
        # net.addLink(sta1, s1, cls=TCLink, bw=v2i_bw[0], delay='10ms')

    # plot
    if '-p' in args:
        net.plotGraph(max_x=400, max_y=1100)

    # configure mobility
    # TODO: make this configurable, different mobility model, etc.
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
    for sta_idx in range(len(stations)):
        stations[sta_idx].setIP('192.168.0.%d/24'%(sta_idx+2), intf='sta%d-eth1'%(sta_idx+1))
        # enable ip forwarding on each sta
        stations[sta_idx].cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    # sta1.setIP('192.168.0.3/24', intf="sta1-eth1")

    server.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')

    info("*** Starting network\n")
    net.build()
    c1.start()
    s1.start([c1])

    # trace replaying, use '-t' for replaying traces
    if '-r' in args:
        for i in range(len(stations)):
            replay_trace_thread_on_sta(stations[i], "sta%d-eth1"%(i+1), v2i_bw_traces[i])
            # replay_trace_thread_on_sta(sta1, "sta1-eth1", v2i_bw_traces[0])


    if '--run_app' in args:
        info("\n*** Running vehicuar server\n")
        if '-f' in args:
            # run server in fix assignemnt mode
            server_cmd = "python3 server.py -f " + assignment_str + "> logs/server.log 2>&1 &"
        else:
            # run server in nomal mode
            server_cmd = "python3 server.py > logs/server.log 2>&1 &"
        server.cmd(server_cmd)
        vehicle_app_commands = []
        tcpdump_cmds = []
        for node_num in range(PATICIPATE_NODES):
            vehicle_app_cmd = 'sleep 8 && python3 vehicle.py %d %s %s > logs/node%d.log 2>&1 &'% \
                                (node_num, vehicle_data_dir[node_num], loc_file, node_num)
            print(vehicle_app_cmd)
            vehicle_app_commands.append(vehicle_app_cmd)
            tcpdump_cmds.append('tcpdump -nni any -s96 -w node%d.pcap &'%node_num)
        # v1_cmd = 'sleep 8 && python3 vehicle.py 0 ../DeepGTAV-data/object-0227-1/ \
        #                 %s > logs/node0.log 2>&1 &'%loc_file
        # v2_cmd = 'sleep 8 && python3 vehicle.py 1 ../DeepGTAV-data/object-0227-1/alt_perspective/0022786/ \
        #                 %s > logs/node1.log 2>&1 &'%loc_file
        # v3_cmd = 'sleep 8 && python3 vehicle.py 2 ../DeepGTAV-data/object-0227-1/alt_perspective/0037122/ \
        #                 %s > logs/node2.log 2>&1 &'%loc_file
        # v4_cmd = 'sleep 8 && python3 vehicle.py 3 ../DeepGTAV-data/object-0227-1/alt_perspective/0191023/ \
        #                 %s > logs/node3.log 2>&1 &'%loc_file
        # v5_cmd = 'sleep 8 && python3 vehicle.py 4 ../DeepGTAV-data/object-0227-1/alt_perspective/0399881/ \
        #                 %s > logs/node4.log 2>&1 &'%loc_file
        # v6_cmd = 'sleep 8 && python3 vehicle.py 5 ../DeepGTAV-data/object-0227-1/alt_perspective/0735239/ \
        #                 %s > logs/node5.log 2>&1 &'%loc_file
        # tcpdump_cmd1 = 'tcpdump -nni any -s96 -w node1.pcap &'
        # execute application and tcpdump commands
        for node_num in range(PATICIPATE_NODES):
            stations[node_num].cmd(vehicle_app_commands[node_num])
            stations[node_num].cmd(tcpdump_cmds[node_num])

    
    # elif '-f' in args:
    #     replay_fixed_assignment(server, stations, assignment_str)
        

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    sta_locs=default_loc
    loc_file=default_loc_file
    assignment_str = None
    start_bandwidth = default_v2i_bw

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
    
    if '--trace' in sys.argv:
        trace_filename = sys.argv[sys.argv.index('--trace')+1]
        all_bandwidth = np.loadtxt(trace_filename)
        start_bandwidth = all_bandwidth[0]
        for i in range(all_bandwidth.shape[1]):
            v2i_bw_traces[i] = all_bandwidth[:, i]


    topology(sys.argv, locations=sta_locs, loc_file=loc_file, \
            assignment_str=assignment_str, v2i_bw=start_bandwidth)  