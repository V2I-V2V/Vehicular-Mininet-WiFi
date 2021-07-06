#!/usr/bin/python3
# Vehicular perception server
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scheduling
import socket
import threading
import time
import ptcl.pcd_merge
import numpy as np
import argparse
import ptcl.pointcloud
import message

MAX_VEHICLES = 8
MAX_FRAMES = 80
TYPE_PCD = 0
TYPE_OXTS = 1
HELPEE = 0
HELPER = 1

sys.stderr = sys.stdout

conn_lock = threading.Lock()


location_map = {}
node_seq_nums = {}
client_sockets = {}
vehicle_types = {} # 0 for helpee, 1 for helper
pcds = [[[] for _ in range(MAX_FRAMES)] for _ in range(MAX_VEHICLES)]
oxts = [[[] for _ in range(MAX_FRAMES)] for _ in range(MAX_VEHICLES)]
curr_processed_frame = 0
data_ready_matrix = np.zeros((MAX_VEHICLES, MAX_FRAMES))
helper_helpee_socket_map = {}
current_assignment = {}
init_time = 0

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_vehicles', default=6, type=int,
                        help='number of vehicles (default: 6)')
parser.add_argument('-f', '--fixed_assignment', nargs='+', type=int, 
                        help='use fixed assignment instead of dynamic shceduling, provide \
                        assignment with spaces (e.g. -f 3 2)')
parser.add_argument('-s', '--scheduler', default='minDist', type=str,
                        help='scheduler to use')
parser.add_argument('-d', '--data_save', default=False, type=bool,
                        help='whether to save undecoded pcds')
args = parser.parse_args()
scheduler_mode = args.scheduler
fixed_assignment = ()
save = args.data_save
if args.fixed_assignment is not None:
    scheduler_mode = "fixed"
    fixed_assignment = scheduling.get_assignment_tuple(args.fixed_assignment)
    print("Run in fix mode")
    print(fixed_assignment)
else:
    print("Run in %s mode" % scheduler_mode)
num_vehicles = args.num_vehicles
 


class SchedThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)


    def run(self):
        global current_assignment
        while True:
            print(location_map, flush=True)
            if len(location_map) == num_vehicles:
                positions = []
                helper_list = []
                for k, v in sorted(location_map.items()):
                    positions.append(v)
                helpee_count = 0
                helper_count = 0
                for k, v in vehicle_types.items():
                    if v == HELPEE:
                        helpee_count += 1
                        helper_list.append(HELPEE)
                    else:
                        helper_count += 1
                        helper_list.append(HELPER)

                if scheduler_mode == 'minDist':
                    # TODO: currently assume all helpees have lower IDs, need to be fixed and made more flexible
                    # TODO: do a transformation to make helpee nodes lower ID
                    assignment = scheduling.min_total_distance_sched(helpee_count, helper_count, positions)
                elif scheduler_mode == 'fixed':
                    # TODO: if fixed assignment, don't need to get vehicles' locations
                    assignment = fixed_assignment
                elif scheduler_mode == 'random':
                    random_seed = (time.time() - init_time) // 5
                    assignment = scheduling.random_sched(helpee_count, helper_count, random_seed)
                print("Assignment: " + str(assignment) + ' ' + str(time.time()))
                # for node_num in range(len(positions)):
                for cnt, node in enumerate(assignment):
                    current_assignment[node] = cnt
                    # print("send %d to node %d" % (cnt, node))
                    msg = cnt.to_bytes(2, 'big')
                    client_sockets[node].send(msg)
                for node_num in range(0, helpee_count+helper_count):
                    if node_num not in assignment:
                        # print("send %d to node %d" % (65535, node_num))
                        msg = int(65535).to_bytes(2, 'big')
                        client_sockets[node_num].send(msg)
            time.sleep(0.2)


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
        client_sockets[vehicle_id] = self.client_socket
        header, payload = message.recv_msg(self.client_socket,\
                                        message.TYPE_CONTROL_MSG)
        while len(header) > 0 and len(payload) > 0:
            payload_size, msg_type = message.parse_control_msg_header(header)
            if msg_type == message.TYPE_LOCATION:
                v_type, v_id, x, y, seq_num = \
                    message.server_parse_location_msg(payload)
                if v_id not in node_seq_nums.keys() or \
                node_seq_nums[v_id] < seq_num:
                    # only update location when seq num is larger
                    print("Recv loc msg with seq num %d from vehicle %d" % (seq_num, v_id))
                    location_map[v_id] = (x, y)
                    vehicle_types[v_id] = v_type
                    node_seq_nums[v_id] = seq_num
            elif msg_type == message.TYPE_ROUTE:
                # implment logic for receiving routing messages
                pass
            header, payload = message.recv_msg(self.client_socket,\
                                        message.TYPE_CONTROL_MSG)
        self.client_socket.close()


def server_recv_data(client_socket, client_addr):
    conn_lock.acquire()
    print("Connect data channel from: ", client_addr)
    data = client_socket.recv(2)
    assert len(data) == 2
    vehicle_id = int.from_bytes(data, "big")
    print("Get sender id %d" % vehicle_id, flush=True)
    conn_lock.release()

    while True:
        t_recv_start = time.time()
        header, msg = message.recv_msg(client_socket, message.TYPE_DATA_MSG)
        if header == b'' and msg == b'':
            print("[Helper relay closed]")
            client_socket.close()
            return
        # v_id is the actual pcd captured vehicle, which might be different from sender vehicle id
        msg_size, frame_id, v_id, data_type = message.parse_data_msg_header(header)
        print("[receive header] frame %d, vehicle id: %d, data size: %d, type: %s" % \
                (frame_id, v_id, msg_size, 'pcd' if data_type == 0 else 'oxts')) 
        assert len(msg) == msg_size
        t_elasped = time.time() - t_recv_start
        if frame_id >= MAX_FRAMES:
            continue
        if data_type == TYPE_PCD:
            print("[Full frame recved] from %d, id %d throughput: %f MB/s %d time: %f" % 
                        (v_id, frame_id, msg_size/1000000.0/t_elasped, msg_size, time.time()), flush=True)
            pcds[v_id][frame_id] = msg
        elif data_type == TYPE_OXTS:
            print("[Oxts recved] from %d, frame id %d" %  (v_id, frame_id))
            oxts[v_id][frame_id] = [float(x) for x in msg.split()]
        
        if len(pcds[v_id][frame_id]) > 0 and len(oxts[v_id][frame_id]) > 0:
            data_ready_matrix[v_id][frame_id] = 1
        

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
            decoded_pcl = ptcl.pointcloud.dracoDecode(pcds[0][curr_processed_frame])
            decoded_pcl = np.append(decoded_pcl, np.zeros((decoded_pcl.shape[0],1),dtype='float32'), axis=1)
            points_oxts_primary = (decoded_pcl, oxts[0][curr_processed_frame])
            points_oxts_secondary = []
            for i in range(1, num_vehicles):
                pcl = ptcl.pointcloud.dracoDecode(pcds[i][curr_processed_frame])
                pcl = np.append(pcl, np.zeros((pcl.shape[0],1), dtype='float32'), axis=1)
                points_oxts_secondary.append((pcl,oxts[i][curr_processed_frame]))
                if save:
                    with open('output/node%d_%d.bin'%(i, curr_processed_frame), 'wb') as f:
                        f.write(pcds[i][curr_processed_frame])
            merged_pcl = ptcl.pcd_merge.merge(points_oxts_primary, points_oxts_secondary)
            # with open('output/merged_%d.bin'%curr_processed_frame, 'w') as f:
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
    global init_time
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
    data_process_thread = threading.Thread(target=merge_data_when_ready)
    data_process_thread.start()
    while True:
        server.listen()
        client_socket, client_address = server.accept()
        newthread = ControlConnectionThread(client_address, client_socket)
        newthread.daemon = True
        newthread.start()


if __name__ == "__main__":
    main()
