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

# global default values
vehicle_data_dir = ['../DeepGTAV-data/object-0227-1/', 
                    '../DeepGTAV-data/object-0227-1/alt_perspective/0022786/',
                    '../DeepGTAV-data/object-0227-1/alt_perspective/0037122/',
                    '../DeepGTAV-data/object-0227-1/alt_perspective/0191023/',
                    '../DeepGTAV-data/object-0227-1/alt_perspective/0399881/',
                    '../DeepGTAV-data/object-0227-1/alt_perspective/0735239/']
default_loc = ['280.225, 891.726, 0', '313.58, 855.46, 0', '286.116, 832.733, 0', \
                '320.134, 854.744, 0', '296.692, 832.28, 0', '290.943, 881.713, 0', \
                '313.943, 891.713, 0', '312.943, 875.713, 0']
default_loc_file="input/object-0227-loc.txt"
default_v2i_bw = [100, 100, 100, 100, 100, 100] # unit: Mbps
v2i_bw_traces = {}
trace_filename = ''


def replay_trace(node, ifname, trace):
    intf = node.intf(ifname)
    time.sleep(8)
    for throughput in trace:
        start_t = time.time()
        if throughput == 0:
            throughput += 0.001
        intf.config(bw=throughput)
        elapsed_t = time.time() - start_t
        sleep_t = 1.0 - elapsed_t
        time.sleep(sleep_t)


def replay_trace_thread_on_sta(sta, ifname, thrpt_trace):
    replay_thread = thread(target=replay_trace, args=(sta, ifname, thrpt_trace))
    replay_thread.daemon = True
    replay_thread.start()


def create_nodes(net, num_nodes, locations):
    server = net.addHost('server', mac='00:00:00:00:00:01', ip='192.168.0.1/24')
    stations = []
    for node_num in range(num_nodes):
        sta = net.addStation('sta%d'%node_num, mac='00:00:00:00:00:%02x'%(node_num+2),\
                            ip6='fe80::%x'%(node_num+1), position=locations[node_num])
        stations.append(sta)
    return server, stations


def create_adhoc_links(net, node, ifname):
    kwargs = dict()
    kwargs['proto'] = 'olsrd'
    channel_num = 5
    net.addLink(node, cls=adhoc, intf=ifname, ssid='adhocNet', \
                mode='g', channel=channel_num, **kwargs)


def create_wired_links(net, node, switch, bw):
    net.addLink(node, switch, cls=TCLink, bw=bw, delay='10ms')


# TODO: think about imlementing this
def config_mobility(net, stations, loc_file):
    pass
    #     net.startMobility(time=0, mob_rep=1, reverse=False)
    #     p1_start, p2_start, p1_end, p2_end = dict(), dict(), dict(), dict()
    #     if '-c' not in args:
    #         p1_start = {'position': '286.116,832.733,0.0'}
    #         p2_start = {'position': '280.225,891.726,0.0'}
    #         p1_end = {'position': '257.905,907.762,0.0'}
    #         p2_end = {'position': '284.816,879.443,0.0'}

    #     net.mobility(stations[0], 'start', time=50, **p1_start)
    #     net.mobility(stations[1], 'start', time=50, **p2_start)
    #     net.mobility(stations[0], 'stop', time=60, **p1_end)
    #     net.mobility(stations[1], 'stop', time=60, **p2_end)
    #     net.stopMobility(time=61)


def setup_ip(node, ip, ifname):
    node.setIP(ip, intf=ifname)
    node.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')


def run_application(server, stations, scheduler, assignment_str):
    num_nodes = len(stations)
    if scheduler == 'fixed':
        # run server in fix assignemnt mode
        server_cmd = "python3 server/server.py -f %s -n %d -t %s > logs/server.log 2>&1 &"\
                        %(assignment_str, num_nodes, trace_filename)
    else:
        # run server in other scheduler mode (minDist, fixed)
        server_cmd = "python3 server/server.py -s %s -n %d -t %s > logs/server.log 2>&1 &"\
                        %(scheduler, num_nodes, trace_filename)
        print(server_cmd)
    server.cmd(server_cmd)
    vehicle_app_commands = []
    for node_num in range(len(stations)):
        vehicle_app_cmd = 'sleep 8 && python3 vehicle/vehicle.py %d %s %s > logs/node%d.log 2>&1 &'% \
                            (node_num, vehicle_data_dir[node_num], loc_file, node_num)
        print(vehicle_app_cmd)
        vehicle_app_commands.append(vehicle_app_cmd)

    # execute application commands
    for node_num in range(len(stations)):
        stations[node_num].cmd(vehicle_app_commands[node_num])


def collect_tcpdump(nodes):
    tcpdump_cmds = []
    for node_num in range(len(nodes)):
        tcpdump_cmds.append('tcpdump -nni any -s96 -w pcaps/node%d.pcap >/dev/null 2>&1 &'%node_num)
        nodes[node_num].cmd(tcpdump_cmds[node_num])


def setup_topology(num_nodes, locations=default_loc, loc_file=default_loc_file, \
                assignment_str=None, v2i_bw=default_v2i_bw, enable_plot=False, \
                enable_tcpdump=False, run_app=False, scheduler="minDist"):
    net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)
    
    info("*** Creating nodes\n")

    server, stations = create_nodes(net, num_nodes, locations)
    s1 = net.addSwitch('s1')
    c1 = net.addController('c1')
    net.setPropagationModel(model="logDistance", exp=4)

    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()

    ### configure adhoc and wired interfaces ###
    info("*** Creating links\n")
    net.addLink(server, s1, cls=TCLink)
    for sta_idx in range(len(stations)):
        create_adhoc_links(net, stations[sta_idx], 'sta%d-wlan0'%sta_idx)
        create_wired_links(net, stations[sta_idx], s1, bw=v2i_bw[sta_idx])

    ### plot if enabled ###
    if enable_plot:
        net.plotGraph(max_x=400, max_y=1100)

    ### configure mobility ###
    # TODO: Implement the following function with different mobility model, etc.
    config_mobility(net, stations, loc_file)

    ### assign ip addresses to wired interfaces ###
    info("*** Addressing...\n")
    server.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    for sta_idx in range(num_nodes):
        setup_ip(stations[sta_idx], '192.168.0.%d/24'%(sta_idx+2), 'sta%d-eth1'%sta_idx)

    info("*** Starting network\n")
    net.build()
    c1.start()
    s1.start([c1])

    ### Trace replaying ###
    for i in range(num_nodes):
        replay_trace_thread_on_sta(stations[i], "sta%d-eth1"%i, v2i_bw_traces[i])

    ### Run application ###
    if run_app is True:
        info("\n*** Running vehicuar server\n")
        run_application(server, stations, scheduler, assignment_str)
    
    ### Collect tcpdump trace ###
    if enable_tcpdump is True:
        info("*** Tcpdump trace enabled\n")
        collect_tcpdump(stations)
        server.cmd('tcpdump -nni any -s96 -w pcaps/server.pcap >/dev/null 2>&1 &')

    info("*** Running CLI\n")
    # CLI(net)
    time.sleep(30)

    info("*** Stopping network\n")
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')

    # take argument number of nodes:
    if '-n' in sys.argv:
        num_nodes = int(sys.argv[sys.argv.index('-n')+1])
        if num_nodes > 6:
            print("Not supported node num, plz use 6 nodes for now!")
            sys.exit()

    ### define default values ###
    sta_locs=default_loc
    loc_file=default_loc_file
    assignment_str = None # only used in fixed assignmented 
    start_bandwidth = [100 for _ in range(num_nodes)]
    enable_plot = False
    enable_tcpdump = False
    run_app = False
    scheduler = 'minDist'

    if '-p' in sys.argv:
        pcd_config_file = sys.argv[sys.argv.index('-p')+1]
        vehicle_data_dir = np.loadtxt(pcd_config_file, dtype=str)

    if '-s' in sys.argv:
        scheduler = sys.argv[sys.argv.index('-s')+1]
        if scheduler == 'fixed':
            print("Run server in fixed mode")
            assignment_filename = sys.argv[sys.argv.index('-s')+2]
            assignment_idx = int(sys.argv[sys.argv.index('-s')+3])
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

    if '--plot' in sys.argv:
        enable_plot = True

    if '--collect-traffic' in sys.argv:
        enable_tcpdump = True

    if '--run_app' in sys.argv:
        run_app = True

    setup_topology(num_nodes, locations=sta_locs, loc_file=loc_file, \
            assignment_str=assignment_str, v2i_bw=start_bandwidth, enable_plot=enable_plot,\
            enable_tcpdump=enable_tcpdump, run_app=run_app, scheduler=scheduler)  