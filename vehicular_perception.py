import sys, os
from numpy.lib import utils
from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mn_wifi.link import wmediumd, adhoc, Intf
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.wmediumdConnector import interference
from mn_wifi.replaying import ReplayingMobility
from threading import Thread as thread
import numpy as np
import time
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import random

# global default values
pcd_data_type = 'GTA'
vehicle_data_dir = ['~/DeepGTAV-data/object-0227-1/', 
                    '~/DeepGTAV-data/object-0227-1/alt_perspective/0022786/',
                    '~/DeepGTAV-data/object-0227-1/alt_perspective/0037122/',
                    '~/DeepGTAV-data/object-0227-1/alt_perspective/0191023/',
                    '~/DeepGTAV-data/object-0227-1/alt_perspective/0399881/',
                    '~/DeepGTAV-data/object-0227-1/alt_perspective/0735239/']
default_loc = ['280.225, 891.726, 0', '313.58, 855.46, 0', '286.116, 832.733, 0', \
                '320.134, 854.744, 0', '296.692, 832.28, 0', '290.943, 881.713, 0', \
                '313.943, 891.713, 0', '312.943, 875.713, 0']
default_loc_file = os.path.dirname(os.path.abspath(__file__)) + "/input/locations/location-multihop.txt"
default_v2i_bw = [100, 100, 100, 100, 100, 100] # unit: Mbps
v2i_bw_traces = {0: [100], 1: [100], 2: [100], 3: [100], 4: [100], 5: [100]}
tx_power = 20
time_to_run = 70
trace_filename = os.path.dirname(os.path.abspath(__file__)) + "/input/traces/constant.txt"
routing = 'olsrd'
no_control = 0
adaptive_encode = 0
adaptive_frame_skip = False
add_noise_to_loc = False
v2v_mode = False
grouping = False
enable_grouping = 0
combine_method = "op_sum"
score_method = "harmonic"
CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def replay_trace(node, ifname, trace):
    intf = node.intf(ifname)
    time.sleep(59)
    print("start update bw:", time.time(), "total length", min(len(trace), time_to_run-20))
    for throughput_idx in range(min(len(trace), time_to_run-20)):
        start_t = time.time()
        if trace[throughput_idx] == 0:
            trace[throughput_idx] += 0.001
        intf.config(bw=trace[throughput_idx])
        elapsed_t = time.time() - start_t
        sleep_t = 1.0 - elapsed_t
        if sleep_t > 0:
            time.sleep(sleep_t)


def replay_trace_thread_on_sta(sta, ifname, thrpt_trace):
    replay_thread = thread(target=replay_trace, args=(sta, ifname, thrpt_trace))
    replay_thread.daemon = True
    replay_thread.start()
    return replay_thread


def create_nodes(net, num_nodes, locations):
    server = net.addHost('server', mac='00:00:00:00:00:01', ip='192.168.0.1/24')
    stations = []
    for node_num in range(num_nodes):
        sta = net.addStation('sta%d'%node_num, mac='00:00:00:00:00:%02x'%(node_num+2),\
                            ip6='fe80::%x'%(node_num+1), position=locations[node_num%len(locations)])
        stations.append(sta)
    return server, stations


def create_adhoc_links(net, node, ifname, channel_num=5, ssid='adhocNet'):
    kwargs = dict()
    if routing == 'olsrd':
        kwargs['proto'] = 'olsrd'
    # channel_num = 5
    net.addLink(node, cls=adhoc, intf=ifname, ssid=ssid, \
                mode='g', channel=channel_num, **kwargs)
    node.setTxPower(tx_power)


def create_wired_links(net, node, switch, bw):
    net.addLink(node, switch, cls=TCLink, bw=bw, delay='10ms')


def read_location_traces(loc_file):
    loc_trace = np.loadtxt(loc_file)
    if loc_trace.ndim == 1:
        loc_trace = loc_trace.reshape(1, -1)
    return loc_trace

# Deprecated, do not use this
def config_mobility_mininet_replay(net, stations, loc_file, plot=True):
    net.isReplaying = True
    loc_trace = read_location_traces(loc_file)
    print('\nUse mininet replay mobility')
    for sta_idx in range(len(stations)):
        stations[sta_idx].p = []
    for time_i in range(loc_trace.shape[0]):
        for sta_idx in range(len(stations)):
            pos = float(loc_trace[time_i][2*sta_idx]), float(loc_trace[time_i][2*sta_idx+1]), 0.0
            stations[sta_idx].p.append(pos)


def config_mobility(net, stations, loc_file, plot=False):
    loc_trace = read_location_traces(loc_file)
    time.sleep(59)
    print("\nstart update location at %f" % time.time())
    # loc_update_logs = []
    # for station_idx in range(len(stations)):
    #     log = open('%d.txt'%station_idx, 'w+')
    #     loc_update_logs.append(log)
    for time_i in range(min(loc_trace.shape[0], time_to_run*10)):
        t_s = time.time()
        for station_idx in range(len(stations)):
            stations[station_idx].setPosition('%f,%f,0'%(loc_trace[time_i][(2*station_idx)%loc_trace.shape[1]], \
                                                     loc_trace[time_i][(2*station_idx+1)%loc_trace.shape[1]]))
            # loc_update_logs[station_idx].write(str(time.time())+' '+str(loc_trace[time_i][2*station_idx])\
            #     + ' ' + str(loc_trace[time_i][2*station_idx+1]) + '\n')
            if enable_plot:
                stations[station_idx].update_2d()
        t_passed = time.time() - t_s
        if t_passed < 0.1:
            time.sleep(0.1-t_passed)
    print("\nfinish update location at %f" % time.time())


def setup_ip(node, ip, ifname):
    node.setIP(ip, intf=ifname)
    node.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    node.cmd('echo 1 > /proc/sys/net/ipv4/tcp_frto')
    # node.cmd('echo 1 > /proc/sys/net/ipv4/tcp_low_latency')


def configure_priority(node, ifname):
    node.cmd('sh prioritize_traffic.sh %s'%ifname)


def kill_application():
    print("*** Stop server.py and vehicle.py ***")
    cmd = "kill -9 $(ps aux | grep \"[v]ehicle.py\" | awk {'print $2'})"
    os.system(cmd)
    cmd = "kill -9 $(ps aux | grep \"[s]erver.py\" | awk {'print $2'})"
    os.system(cmd)
    cmd = "kill -9 $(ps aux | grep \"[d]ynamic.py\" | awk {'print $2'})"
    os.system(cmd)


def run_application(server, stations, scheduler, assignment_str, helpee_conf=None, fps=1,\
                    save=0, is_one_to_many=1, is_v2v_mode=False):
    num_nodes = len(stations)
    if scheduler == 'carspeak':
        vehicle_app_commands = []
        for node_num in range(len(stations)):
            vehicle_app_cmd = 'python3 -u %s/vehicle/carspeak.py -i %d -d %s -l %s -f %d --start_timestamp %f'\
                % (CODE_DIR, node_num, vehicle_data_dir[node_num%len(vehicle_data_dir)], loc_file, fps, time.time()+5)
            vehicle_app_cmd += ' --data_type %s > %s/logs/node%d.log 2>&1 &'%(pcd_data_type, CODE_DIR, node_num)
            print(vehicle_app_cmd)
            vehicle_app_commands.append(vehicle_app_cmd)
        for node_num in range(len(stations)):
            # cmd_start = time.time()
            stations[node_num].cmd(vehicle_app_commands[node_num])
        return   
    if scheduler == 'fixed':
        # run server in fix assignemnt mode
        server_cmd = "python3 -u %s/server/server.py -f %s -n %d -t %s -d %d -m %d --data_type %s --v2v_mode %d > %s/logs/server.log 2>&1 &"\
             % (CODE_DIR, assignment_str, num_nodes, trace_filename, save, is_one_to_many, pcd_data_type, int(is_v2v_mode), CODE_DIR)
    else:
        # run server in other scheduler mode (minDist, fixed)
        server_cmd = "python3 -u %s/server/server.py -s %s -n %d -t %s -d %d -m %d --data_type %s --combine_method %s --score_method %s --v2v_mode %d --enable_grouping %d > %s/logs/server.log 2>&1 &"\
             % (CODE_DIR, scheduler, num_nodes, trace_filename, save, is_one_to_many, pcd_data_type, combine_method, score_method, int(is_v2v_mode), enable_grouping, CODE_DIR)
        print(server_cmd)
    if is_v2v_mode:
        # rand_vehicle_server = random.randint(0, len(stations)-1)
        stations[0].cmd(server_cmd)
        if len(stations) > 10:
            v2v_server_cmd = "python3 -u %s/server/server.py -s %s -n %d -t %s -d %d -m %d --data_type %s --combine_method %s --score_method %s --v2v_mode %d --enable_grouping %d > %s/logs/V2V_server.log 2>&1 &"\
             % (CODE_DIR, scheduler, num_nodes, trace_filename, save, is_one_to_many, pcd_data_type, combine_method, score_method, int(is_v2v_mode), enable_grouping, CODE_DIR)
            stations[10].cmd(v2v_server_cmd)            
    else:
        server.cmd(server_cmd)
        # Use a backup V2V server
        v2v_server_cmd = "python3 -u %s/server/server.py -s %s -n %d -t %s -d %d -m %d --data_type %s --combine_method %s --score_method %s --v2v_mode %d > %s/logs/v2v_server.log 2>&1 &"\
                % (CODE_DIR, scheduler, num_nodes, trace_filename, save, is_one_to_many, pcd_data_type, combine_method, score_method, 1, CODE_DIR)    
        stations[0].cmd(v2v_server_cmd)    

    time.sleep(4)
    vehicle_app_commands = []
    for node_num in range(len(stations)):
        vehicle_app_cmd = 'python3 -u %s/vehicle/vehicle.py -i %d -d %s -l %s -c %s -f %d -n %d --adaptive %d --v2v_mode %d --start_timestamp %f'\
            % (CODE_DIR, node_num, vehicle_data_dir[node_num%len(vehicle_data_dir)], loc_file, helpee_conf, fps, no_control, adaptive_encode, int(is_v2v_mode), time.time()+5)
        if adaptive_frame_skip:
            vehicle_app_cmd += ' --adapt_skip_frames '
        if add_noise_to_loc:
            vehicle_app_cmd += ' --add_loc_noise '
        vehicle_app_cmd += ' --data_type %s > %s/logs/node%d.log 2>&1 &'%(pcd_data_type, CODE_DIR, node_num)
        print(vehicle_app_cmd)
        vehicle_app_commands.append(vehicle_app_cmd)

    # execute application commands
    for node_num in range(len(stations)):
        # cmd_start = time.time()
        stations[node_num].sendCmd(vehicle_app_commands[node_num])
        # print('cmd taking', time.time() - cmd_start)


def collect_tcpdump(nodes):
    tcpdump_cmds = []
    for node_num in range(len(nodes)):
        tcpdump_cmds.append('tcpdump -nni any -s96 -w %s/pcaps/node%d.pcap >/dev/null 2>&1 &'%\
            (CODE_DIR, node_num))
        nodes[node_num].cmd(tcpdump_cmds[node_num])


def run_custom_routing(nodes):
    routing_cmds = []
    for node_num in range(len(nodes)):
        routing_cmds.append('sleep 2 && python3 %s/routing/dynamic.py %d > %s/logs/node%d.route 2>&1 &'\
                            %(CODE_DIR, node_num, CODE_DIR, node_num))
        nodes[node_num].cmd(routing_cmds[node_num])


def setup_topology(num_nodes, locations=default_loc, loc_file=default_loc_file, \
                assignment_str=None, v2i_bw=default_v2i_bw, enable_plot=False, \
                enable_tcpdump=False, run_app=False, scheduler="minDist",
                helpee_conf=None, fps=1, save=0, mininet_replay_mob=False, is_one_to_many=1,
                combine_method="op_sum", score_method="harmonic"):
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
        if not grouping:
            create_adhoc_links(net, stations[sta_idx], 'sta%d-wlan0'%sta_idx)
        else:
            if not v2v_mode:
                if sta_idx in [0, 1, 4, 8, 10, 11, 14, 18]:
                    create_adhoc_links(net, stations[sta_idx], 'sta%d-wlan0'%sta_idx)
                else:
                    create_adhoc_links(net, stations[sta_idx], 'sta%d-wlan0'%sta_idx, channel_num=11, ssid='v2v-group2')
            else:
                if sta_idx < 10:
                    create_adhoc_links(net, stations[sta_idx], 'sta%d-wlan0'%sta_idx)
                else:
                    create_adhoc_links(net, stations[sta_idx], 'sta%d-wlan0'%sta_idx, channel_num=11, ssid='v2v-group2')
                    
        create_wired_links(net, stations[sta_idx], s1, bw=v2i_bw[sta_idx%len(v2i_bw)])

    ### plot if enabled ###
    if enable_plot:
        net.plotGraph(max_x=400, max_y=1100)

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
    replaying_threads = []
    # if not v2v_mode:
    # it should not matter if we replay trace on v2v mode
    for i in range(num_nodes):
        replaying_thread = replay_trace_thread_on_sta(stations[i], "sta%d-eth1"%i, v2i_bw_traces[i%len(v2i_bw_traces.keys())])
        replaying_threads.append(replaying_thread)


    ### Configure routing if custom
    if routing == 'custom':
        run_custom_routing(stations)

    # ### Prioritize Traffic ### 
    for sta_idx in range(num_nodes):
        configure_priority(stations[sta_idx], 'sta%d-wlan0'%sta_idx)
        # stations[sta_idx].cmd('iwconfig sta%d-wlan0 retry limit 15'%sta_idx)

    ### Run application ###
    if run_app is True:
        info("\n*** Running vehicular server\n")
        run_application(server, stations, scheduler, assignment_str, helpee_conf, fps, save, is_one_to_many, v2v_mode)
    
    ### Collect tcpdump trace ###
    if enable_tcpdump is True:
        info("*** Tcpdump trace enabled\n")
        collect_tcpdump(stations)
        server.cmd('tcpdump -nni any -s96 -w pcaps/server.pcap >/dev/null 2>&1 &')

    ### configure mobility ###
    # TODO: Implement the following function with different mobility model, etc.
    if mininet_replay_mob:
        config_mobility_mininet_replay(net, stations, loc_file)
        ReplayingMobility(net)
    else:
        mobility_thread = thread(target=config_mobility, args=(net, stations, loc_file, enable_plot))
        mobility_thread.start()

    info("*** Running CLI\n")
    if run_app:
        start = time.time()
        print("--- start sleep", start)
        time.sleep(time_to_run)
        print("--- stop", time.time() - start)
    else:
        CLI(net)

    mobility_thread.join()
    for replaying_thread in replaying_threads:
        replaying_thread.join()
    
    kill_application()

    info("*** Stopping network\n")
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')

    # take argument number of nodes:
    if '-n' in sys.argv:
        num_nodes = int(sys.argv[sys.argv.index('-n')+1])
        if num_nodes > 100 and '--run_app' in sys.argv:
            print("Not supported node num to run application, plz use 6 nodes for now!")
            sys.exit()

    ### define default values ###
    sta_locs=default_loc
    loc_file=default_loc_file
    assignment_str = None # only used in fixed assignmented 
    start_bandwidth = [100 for _ in range(num_nodes)]
    enable_plot = False
    enable_tcpdump = False
    run_app = False
    data_save = 0 # by default, dont save pcd
    is_one_to_many = 1
    scheduler = 'minDist'
    fps = 1
    helpee_conf_file = os.path.dirname(os.path.abspath(__file__)) + '/input/helpee_conf/helpee-nodes.txt'
    mininet_mob_replay = False

    if '--replay-mobility' in sys.argv:
        mininet_mob_replay = True

    if '-p' in sys.argv:
        pcd_config_file = sys.argv[sys.argv.index('-p')+1]
        vehicle_data_config = np.loadtxt(pcd_config_file, dtype=str)
        pcd_data_type = vehicle_data_config[0]
        vehicle_data_dir = vehicle_data_config[1:]

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
        loc_file = loc_filename
        locs = np.loadtxt(loc_filename)
        if locs.ndim > 1:
            locs = locs[0]
        sta_locs = utils.produce_3d_location_arr(locs)
        print(sta_locs)
    
    if '--trace' in sys.argv:
        trace_filename = sys.argv[sys.argv.index('--trace')+1]
        all_bandwidth = np.loadtxt(trace_filename)
        start_bandwidth = all_bandwidth[0]
        for i in range(num_nodes):
            v2i_bw_traces[i] = all_bandwidth[:, i%all_bandwidth.shape[1]]
        # for i in range(all_bandwidth.shape[1]):
        #     v2i_bw_traces[i] = all_bandwidth[:, i]

    if '--plot' in sys.argv:
        enable_plot = True

    if '--collect-traffic' in sys.argv:
        enable_tcpdump = True

    if '--run_app' in sys.argv:
        run_app = True
        if '--fps' in sys.argv:
            fps = int(sys.argv[sys.argv.index('--fps')+1])

    if '--helpee_conf' in sys.argv:
        helpee_conf_file = sys.argv[sys.argv.index('--helpee_conf')+1]
    
    if '--save_data' in sys.argv:
        data_save = 1

    if '-t' in sys.argv:
        time_to_run = int(sys.argv[sys.argv.index('-t')+1])

    if '--no_control' in sys.argv:
        no_control = 1
    
    if '-r' in sys.argv:
        routing = sys.argv[sys.argv.index('-r')+1]
    
    if '--power' in sys.argv:
        tx_power = int(sys.argv[sys.argv.index('--power')+1])

    if '--multi' in sys.argv:
        is_one_to_many = int(sys.argv[sys.argv.index('--multi') + 1])
    
    if '--adaptive_encode' in sys.argv:
        adaptive_encode = int(sys.argv[sys.argv.index('--adaptive_encode')+1])
    
    if '--adapt_frame_skipping' in sys.argv:
        adaptive_frame_skip = True
    
    if '--combine_method' in sys.argv:
        combine_method = sys.argv[sys.argv.index('--combine_method') + 1]
    
    if '--add_noise_to_loc' in sys.argv:
        add_noise_to_loc = True
        
    if '--score_method' in sys.argv:
        score_method = sys.argv[sys.argv.index('--score_method') + 1]
    
    if '--v2v_mode' in sys.argv:
        v2v_mode = True
    
    if '--group' in sys.argv:
        grouping = True
        enable_grouping = 1

    setup_topology(num_nodes, locations=sta_locs, loc_file=loc_file, \
            assignment_str=assignment_str, v2i_bw=start_bandwidth, enable_plot=enable_plot,\
            enable_tcpdump=enable_tcpdump, run_app=run_app, scheduler=scheduler,
            helpee_conf=helpee_conf_file, fps=fps, save=data_save,\
            mininet_replay_mob=mininet_mob_replay, is_one_to_many=is_one_to_many, combine_method=combine_method, score_method=score_method) 
