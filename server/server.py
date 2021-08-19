#!/usr/bin/python3
# Vehicular perception server
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stderr = sys.stdout
import scheduling
import socket
import threading
import time
import ptcl.pcd_merge
import numpy as np
import argparse
import ptcl.pointcloud
import network.message
import config

MAX_VEHICLES = 8
MAX_FRAMES = 80
TYPE_PCD = 0
TYPE_OXTS = 1
HELPEE = 0
HELPER = 1
NODE_LEFT_TIMEOUT = 0.5
SCHED_PERIOD = 0.2
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.stderr = sys.stdout

curr_connected_vehicles = 0
conn_lock = threading.Lock()
data_save_lock = threading.Lock()
data_save_cv = threading.Condition()
init_time = 0
bws = {0: 50, 1:50, 2:50, 3:50, 4:50, 5:50}


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_vehicles', default=6, type=int, 
                        help='number of vehicles (default: 6)')
parser.add_argument('-f', '--fixed_assignment', nargs='+', type=int, 
                        help='use fixed assignment instead of dynamic shceduling, provide \
                        assignment with spaces (e.g. -f 3 2)')
parser.add_argument('-s', '--scheduler', default='minDist', type=str, 
                    help='scheduler to use: minDist|random|bwAware')
parser.add_argument('-t', '--trace_filename', default='', type=str, help='trace file to use')
parser.add_argument('-d', '--data_save', default=0, type=int, 
                    help='whether to save undecoded pcds')
parser.add_argument('-m', '--multi', default=1, type=int, 
                    help='whether to use one-to-many assignment')
parser.add_argument('--data_type', default="GTA", choices=["GTA", "Carla"])
parser.add_argument('--combine_method', default="harmonic_sum", choices=["harmonic_sum", "sum_min"])

args = parser.parse_args()
trace_filename = args.trace_filename
scheduler_mode = args.scheduler
pcd_data_type = args.data_type
fixed_assignment = ()
save = args.data_save
num_vehicles = args.num_vehicles
is_one_to_one = 1 - args.multi
combine_method = args.combine_method
if args.fixed_assignment is not None:
    scheduler_mode = "fixed"
    fixed_assignment = scheduling.get_assignment_tuple(args.fixed_assignment)
    print("Run in fix mode")
    print(fixed_assignment)
else:
    print("Run in %s mode" % scheduler_mode)

all_bandwidth = np.loadtxt(trace_filename)
location_map = {}
route_map = {} # map v_id to the vechile's routing table
node_seq_nums = {}
node_last_recv_timestamp = {}
client_sockets = {}
vehicle_types = {} # 0 for helpee, 1 for helper
received_bytes = {}
node_latency = {}
current_connected_vids = []
pcds = [[[] for _ in range(MAX_FRAMES)] for _ in range(MAX_VEHICLES)]
oxts = [[[] for _ in range(MAX_FRAMES)] for _ in range(MAX_VEHICLES)]
curr_processed_frame = 0
data_ready_matrix = np.zeros((MAX_VEHICLES, MAX_FRAMES))
helper_helpee_socket_map = {}
current_assignment = {}

def get_bws():
    curr_time = time.time()
    if curr_time - init_time < 13: # sync with mininet bw update
        bw_idx = 0  
    elif curr_time - init_time > all_bandwidth.shape[0]:
        bw_idx = int(all_bandwidth.shape[0] - 1)
    else:
        bw_idx = int(curr_time - init_time - 13)
    for i in range(num_vehicles):
        bws[i] = all_bandwidth[bw_idx][i]


def get_mapped_bw(mapped_node_ids):
    get_bws()
    mapped_bw = []
    for node in mapped_node_ids:
        mapped_bw.append(bws[node])
    return mapped_bw


class SchedThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.flip_cnt = 0
        self.last_assignment_score = 0 # used for combined sched
        self.last_assignment = None
        self.assignment_change_threshold = 0.4


    def check_if_loc_map_complete(self, vids):
        return set(vids).issubset(set(location_map.keys()))
    
    def check_if_route_map_conplete(self, vids):
        return set(vids).issubset(set(route_map.keys()))


    def ready_to_schedule(self, scheduler_mode):
        global current_connected_vids
        current_connected_vids = check_current_connected_vehicles()
        print("Current connected vehicles " + str(current_connected_vids))
        helpee_count, helper_count = 0, 0
        for i in current_connected_vids:
            if vehicle_types[i] == HELPEE:
                helpee_count += 1
            else:
                helper_count += 1
        print("Helpers: %d helpees: %d %f"%(helper_count, helpee_count, time.time()))
        if len(current_connected_vids) < 2:
            return False
        elif scheduler_mode == "minDist" or scheduler_mode == "bwAware" or scheduler_mode == 'random' \
            or scheduler_mode == 'switch':
            return self.check_if_loc_map_complete(current_connected_vids)
        elif scheduler_mode == 'routeAware':
            return self.check_if_route_map_conplete(current_connected_vids)
        elif scheduler_mode == 'combined':
            return self.check_if_loc_map_complete(current_connected_vids) and \
                self.check_if_route_map_conplete(current_connected_vids)
        elif scheduler_mode == 'fixed':
            return True
        else:
            return False


    def run(self):
        global current_assignment
        while True:
            sched_start_t = time.time()
            skip_sending_assignment = False
            print("loc:" + str(location_map), flush=True)
            if scheduler_mode == 'fixed':
                assignment = fixed_assignment
                print("Assignment: " + str(assignment) + ' ' + str(time.time()))
                for cnt, node in enumerate(assignment):
                    real_helpee, real_helper = cnt, node
                    current_assignment[real_helpee] = real_helper
                    print("send %d to node %d" % (real_helpee, real_helper))
                    msg = int(real_helpee).to_bytes(2, 'big')
                    if real_helper in client_sockets.keys():
                        client_sockets[real_helper].send(msg)
            elif self.ready_to_schedule(scheduler_mode):
                # print(scheduler_mode)
                positions = []
                routing_tables = {}
                helper_list = []
                helpee_count, helper_count = 0, 0
                for i in current_connected_vids:
                    if vehicle_types[i] == HELPEE:
                        helpee_count += 1
                        helper_list.append(HELPEE)
                    else:
                        helper_count += 1
                        helper_list.append(HELPER)
                if helpee_count == 0: # skip scheduling if no helpee
                    continue
                helper_list = np.array(helper_list)
                mapped_nodes = np.array(current_connected_vids)[np.argsort(helper_list)]
                print("mapped nodes " + str(mapped_nodes))
                original_to_new = {}
                for cnt, mapped_node_id in enumerate(mapped_nodes):
                    original_to_new[mapped_node_id] = cnt
                    if scheduler_mode == 'combined' or scheduler_mode == 'minDist' or \
                        scheduler_mode == 'bwAware':
                        positions.append(location_map[mapped_node_id])
                if scheduler_mode == 'combined' or scheduler_mode == 'routeAware':
                    # print("[mapping routing tables]")
                    for cnt, mapped_node_id in enumerate(mapped_nodes):
                        routing_table = {}
                        if mapped_node_id in route_map.keys():
                            for k, v in route_map[mapped_node_id].items():
                                if v in original_to_new.keys() and k in original_to_new.keys(): # helpee/helper num may change
                                    routing_table[original_to_new[k]] = original_to_new[v]
                                    # print("update routing table", original_to_new[k], original_to_new[v])
                        # routing_tables[original_to_new[k]] = routing_table[original_to_new[k]]
                        routing_tables[cnt] = routing_table
                sched_start = time.time()
                if scheduler_mode == 'combined':
                    # get_bws() # need to map here
                    bws = get_mapped_bw(mapped_nodes)
                    assignment, score, scores = scheduling.combined_sched(helpee_count, helper_count, positions, bws, routing_tables, is_one_to_one, combine_method)
                    if self.last_assignment is not None:
                        last_assignment_id = scheduling.get_id_from_assignment(self.last_assignment)
                    else:
                        last_assignment_id = ()
                    last_score = scores[last_assignment_id] if last_assignment_id in scores.keys() else 0
                    print("best score: ", score, last_score, self.last_assignment_score)
                    if score < last_score + self.assignment_change_threshold:
                        print("Skip assignment ", score, self.last_assignment_score, current_assignment)
                        skip_sending_assignment = True
                    # if score < self.last_assignment_score + self.assignment_change_threshold and \
                    #     score > self.last_assignment_score:
                    #     print("Skip assignment ", score, self.last_assignment_score, current_assignment)
                    #     print("Assignment: " + str(self.last_assignment) + ' ' + str(mapped_nodes) +  ' ' + str(time.time()))
                    #     time.sleep(0.2)
                    #     continue
                    else:
                        self.last_assignment_score = score
                elif scheduler_mode == 'minDist':
                    assignment = scheduling.min_total_distance_sched(helpee_count, helper_count, positions, is_one_to_one)
                elif scheduler_mode == 'bwAware':
                    # get_bws()
                    bws = get_mapped_bw(mapped_nodes)
                    assignment = scheduling.wwan_bw_sched(helpee_count, helper_count, bws, is_one_to_one)
                elif scheduler_mode == 'routeAware':
                    # print("unmapped:", route_map)
                    # print("routing tables", routing_tables)
                    assignment = scheduling.route_sched(helpee_count, helper_count, routing_tables, is_one_to_one)
                elif scheduler_mode == 'random':
                    random_seed = (time.time() - init_time) // 5
                    assignment = scheduling.random_sched(helpee_count, helper_count, random_seed, is_one_to_one)
                elif scheduler_mode == 'switch':
                    if self.flip_cnt % 2 == 0:
                        assignment = (1,)
                    else:
                        assignment = (2,)
                    self.flip_cnt += 1
                
                sched_end = time.time()
                print("Sched takes " + str(sched_end-sched_start))
                if skip_sending_assignment:
                    print("Assignment: " + str(self.last_assignment) + ' ' + str(mapped_nodes) +  ' ' + str(time.time()))
                if not skip_sending_assignment:
                    print("Assignment: " + str(assignment) + ' ' + str(mapped_nodes) +  ' ' + str(time.time()))
                    self.last_assignment = assignment
                    real_helpers = []
                    for cnt, node in enumerate(assignment):
                        real_helpee, real_helper = mapped_nodes[cnt], mapped_nodes[node]
                        current_assignment[real_helpee] = real_helper
                        print("send %d to node %d" % (real_helpee, real_helper))
                        msg = int(real_helpee).to_bytes(2, 'big')
                        client_sockets[real_helper].send(msg)
                        real_helpers.append(real_helper)
                    # for node_id, is_helper in enumerate(helper_list):
                    #     if is_helper == 1 and node_id not in real_helpers:
                            
                    #         msg = int(65535).to_bytes(2, 'big')
                    #         client_sockets[node_id].send(msg)
            sched_elapsed_t = time.time() - sched_start_t
            print("One round sched takes", sched_elapsed_t)
            time.sleep(SCHED_PERIOD)


class ControlConnectionThread(threading.Thread):

    def __init__(self, client_address, client_socket):
        threading.Thread.__init__(self)
        self.client_socket = client_socket
        self.client_address = client_address
        print("New control channel added: ", client_address)


    def run(self):
        # print("Connection from : ", self.client_address)
        data = self.client_socket.recv(2)
        vehicle_id = int.from_bytes(data, "big")
        # send back the shced scheme
        encoded_sched_scheme = config.map_scheduler_to_int_encoding[scheduler_mode].to_bytes(2, 'big')
        self.client_socket.send(encoded_sched_scheme)
        vehicle_types[vehicle_id] = HELPER
        # node_last_recv_timestamp[vehicle_id] = time.time()
        client_sockets[vehicle_id] = self.client_socket
        header, payload = network.message.recv_msg(self.client_socket,\
                                        network.message.TYPE_CONTROL_MSG)
        while len(header) > 0 and len(payload) > 0:
            payload_size, msg_type = network.message.parse_control_msg_header(header)
            if msg_type == network.message.TYPE_LOCATION:
                v_type, v_id, x, y, seq_num = \
                    network.message.server_parse_location_msg(payload)
                if v_id not in node_seq_nums.keys() or \
                node_seq_nums[v_id] < seq_num:
                    # only update location when seq num is larger
                    print("Recv loc msg with seq num %d from vehicle %d" % (seq_num, v_id))
                    location_map[v_id] = (x, y)
                    vehicle_types[v_id] = v_type
                    node_seq_nums[v_id] = seq_num
            elif msg_type == network.message.TYPE_ROUTE:
                v_type, v_id, routing_table, seq_num = network.message.server_parse_route_msg(payload)
                route_map[v_id] = routing_table
                print(route_map)
                vehicle_types[v_id] = v_type
                node_seq_nums[v_id] = seq_num
            node_last_recv_timestamp[v_id] = time.time()
            header, payload = network.message.recv_msg(self.client_socket, network.message.TYPE_CONTROL_MSG)
        self.client_socket.close()


def check_current_connected_vehicles():
    curr_connected_vehicles = []
    for v_id, last_ts in sorted(node_last_recv_timestamp.items()):
        curr_ts = time.time()
        if curr_ts - last_ts < NODE_LEFT_TIMEOUT: 
            # node are still reachable from the server, either helpee or helper
            curr_connected_vehicles.append(v_id)
    return curr_connected_vehicles


def throughput_calc():
    time.sleep(6)
    while True:
        for k, v in received_bytes.items():
            thrpt = v*8.0/1000000
            received_bytes[k] = 0
            print("[Node %d thrpt] %f" %(k, thrpt))
        time.sleep(1.0)


def update_node_latency_dict(node_id, latency):
    if node_id in node_latency.keys():
        node_latency[node_id].append(latency)
    else:
        node_latency[node_id] = [latency]


def server_recv_data(client_socket, client_addr):
    conn_lock.acquire()
    print("Connect data channel from: ", client_addr)
    data = client_socket.recv(2)
    assert len(data) == 2
    vehicle_id = int.from_bytes(data, "big")
    if vehicle_id not in received_bytes.keys():
        received_bytes[vehicle_id] = 0
    print("Get sender id %d" % vehicle_id, flush=True)
    conn_lock.release()

    while True:
        header, msg, throughput, elapsed_t = network.message.recv_msg(client_socket, network.message.TYPE_DATA_MSG)
        if header == b'' and msg == b'':
            print("[Helper relay closed]")
            client_socket.close()
            return
        # v_id is the actual pcd captured vehicle, which might be different from sender vehicle id
        msg_size, frame_id, v_id, data_type, ts, num_chunks, chunk_sizes = network.message.parse_data_msg_header(header)
        # received_bytes[vehicle_id] += msg_size
        # print("[receive header] frame %d, vehicle id: %d, data size: %d, type: %s" % \
        #         (frame_id, v_id, msg_size, 'pcd' if data_type == 0 else 'oxts')) 
        # assert len(msg) == msg_size
        latency = time.time() - ts
        print("[receive header] node %d latency %f" % (v_id, latency))
        # node_last_recv_timestamp[v_id] = time.time()
        # if frame_id >= MAX_FRAMES:
        #     continue
        if data_type == TYPE_PCD:
            print("[Full frame recved] from %d, id %d throughput: %f Mbps %f %d time: %f" % 
                        (v_id, frame_id, throughput, elapsed_t, msg_size, time.time()), flush=True)
            # send back a ACK back
            try:
                client_socket.send(frame_id.to_bytes(2, 'big'))
            except:
                print("[Helper relay closed]")
            update_node_latency_dict(v_id, latency)    
            pcds[v_id][frame_id%MAX_FRAMES] = msg
            if save:
                saving_thread = threading.Thread(target=save_ptcl, args=(v_id, frame_id, msg, num_chunks, chunk_sizes))
                saving_thread.daemon = True
                saving_thread.start()
        elif data_type == TYPE_OXTS:
            # print("[Oxts recved] from %d, frame id %d" %  (v_id, frame_id))
            if pcd_data_type == 'GTA':
                oxts[v_id][frame_id%MAX_FRAMES] = [float(x) for x in msg.split()]
            elif pcd_data_type == 'Carla':
                oxts[v_id][frame_id%MAX_FRAMES] = np.frombuffer(msg).reshape(4,4)
        
        if len(pcds[v_id][frame_id%MAX_FRAMES]) > 0 and len(oxts[v_id][frame_id%MAX_FRAMES]) > 0:
            # if (pcd_data_type == 'GTA' and len(oxts[v_id][frame_id%MAX_FRAMES]) > 0) or \
            #     pcd_data_type == 'Carla':
                # if dataset is GTA, require additional oxts file
            data_ready_matrix[v_id][frame_id%MAX_FRAMES] = 1


def save_ptcl(v_id, frame_id, data, num_chunks, chunk_sizes):
    chunk_num = 0
    if num_chunks == 1:
        with open('%s/output/node%d_frame%d_chunk%d.bin'%(REPO_DIR, v_id, frame_id, chunk_num), 'wb') as f:
            f.write(data)
            f.close()
    else:
        for i in range(num_chunks):
            if i > 0:
                chunk = data[sum(chunk_sizes[:i]):sum(chunk_sizes[:i+1])]
            else:
                chunk = data[:chunk_sizes[0]]
            with open('%s/output/node%d_frame%d_chunk%d.bin'%(REPO_DIR, v_id, frame_id, chunk_num), 'wb') as f:
                f.write(chunk)
                f.close()
            chunk_num += 1

def merge_data_when_ready():
    global curr_processed_frame
    while curr_processed_frame < MAX_FRAMES:
        ready = True
        for n in range(num_vehicles):
            if not data_ready_matrix[n][curr_processed_frame]:
                ready = False
                break
        if ready:
            print("[merge data] merge frame %d at %f" % (curr_processed_frame, time.time()))
            if pcd_data_type == "GTA":
                decoded_pcl = ptcl.pointcloud.dracoDecode(pcds[0][curr_processed_frame])
                # add a redudant intensity value column
                decoded_pcl = np.append(decoded_pcl, np.zeros((decoded_pcl.shape[0],1),dtype='float32'), axis=1)
                points_oxts_primary = (decoded_pcl, oxts[0][curr_processed_frame])
                points_oxts_secondary = []
                for i in range(1, num_vehicles):
                    pcl = ptcl.pointcloud.dracoDecode(pcds[i][curr_processed_frame])
                    if pcl.shape[0] != 0:
                        pcl = np.append(pcl, np.zeros((pcl.shape[0],1), dtype='float32'), axis=1)
                        points_oxts_secondary.append((pcl,oxts[i][curr_processed_frame]))
                merged_pcl = ptcl.pcd_merge.merge(points_oxts_primary, points_oxts_secondary)
            elif pcd_data_type == "Carla":
                decoded_pcls, ptcl_oxts = [], []
                for i in range(0, num_vehicles):
                    pcl = ptcl.pointcloud.dracoDecode(pcds[i][curr_processed_frame])
                    if pcl.shape[0] != 0:
                        decoded_pcls.append(pcl)
                        ptcl_oxts.append(oxts[i][curr_processed_frame])
                merged_pcl = ptcl.pcd_merge.merge_carla(decoded_pcls, ptcl_oxts)
            # with open('%s/output/merged_%d.bin'%(REPO_DIR, curr_processed_frame), 'w') as f:
            #     merged_pcl.tofile(f)
            curr_processed_frame += 1


class DataConnectionThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        HOST = ''
        PORT = 6667
        self.data_channel_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_channel_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.data_channel_sock.bind((HOST, PORT))
        print("Waiting for client data")
    
    def run(self):
        while True:
            self.data_channel_sock.listen()
            client_socket, client_address = self.data_channel_sock.accept()
            new_data_recv_thread = threading.Thread(target=server_recv_data, \
                                                    args=(client_socket,client_address))
            new_data_recv_thread.start()


def main():
    global init_time, curr_connected_vehicles
    init_time = time.time()
    HOST = ''
    PORT = 6666
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    print("Vehicular perception server started", flush=True)
    print("Waiting for client request..", flush=True)
    sched_thread = SchedThread()
    sched_thread.daemon = True
    sched_thread.start()
    data_channel_thread = DataConnectionThread()
    data_channel_thread.daemon = True
    data_channel_thread.start()
    # data_process_thread = threading.Thread(target=merge_data_when_ready)
    # data_process_thread.deamon = True
    # data_process_thread.start()
    # thrpt_calc_thread = threading.Thread(target=throughput_calc)
    # thrpt_calc_thread.start()
    while True:
        server.listen()
        client_socket, client_address = server.accept()
        curr_connected_vehicles += 1 # add a new vehicle to schedule
        newthread = ControlConnectionThread(client_address, client_socket)
        newthread.daemon = True
        newthread.start()


if __name__ == "__main__":
    main()
