#!/usr/bin/python3
# Vehicular perception server

import scheduling
import socket
import threading
import time
import pcd_merge
import numpy as np
import argparse

MAX_VEHICLES = 8
MAX_FRAMES = 80
TYPE_PCD = 0
TYPE_OXTS = 1

print("running server", flush=True)
curr_vehicles = 6
location_map = {}
client_sockets = {}
client_data_sockets = {}
vehicle_types = {} # 0 for helpee, 1 for helper
vehicle_pcds = {}
pcds = [[[] for _ in range(MAX_FRAMES)] for _ in range(MAX_VEHICLES)]
oxts = [[[] for _ in range(MAX_FRAMES)] for _ in range(MAX_VEHICLES)]
curr_processed_frame = 0
data_ready_matrix = np.zeros((MAX_VEHICLES, MAX_FRAMES))
helper_helpee_socket_map = {}
current_assignment = {}


class SchedThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)


    def run(self):
        global current_assignment
        while True:
            # print(time.time())
            print(location_map, flush=True)
            if len(location_map) == 6:
                positions = []
                for k, v in sorted(location_map.items()):
                    positions.append(v)
                helpee_count = 0
                helper_count = 0
                for k, v in vehicle_types.items():
                    if v == 0:
                        helpee_count += 1
                    else:
                        helper_count += 1

                assignment = scheduling.min_total_distance_sched(helpee_count, helper_count, positions)
                print(assignment)
                # for node_num in range(len(positions)):
                #     if node_num in assignment:
                for cnt, node in enumerate(assignment):
                    current_assignment[node] = cnt
                    print("send %d to node %d" % (cnt, node))
                    # print(cnt, node, client_sockets[node])
                    msg = cnt.to_bytes(2, 'big')
                    client_sockets[node].send(msg)
                for node_num in range(0, helpee_count+helper_count):
                    if node_num not in assignment:
                        print("send %d to node %d" % (65535, node_num))
                        msg = int(65535).to_bytes(2, 'big')
                        client_sockets[node_num].send(msg)
            time.sleep(0.2)


class ConnectionThread(threading.Thread):

    def __init__(self, client_address, client_socket):
        threading.Thread.__init__(self)
        self.client_socket = client_socket
        self.client_address = client_address
        print("New connection added: ", client_address)


    def run(self):
        print("Connection from : ", self.client_address)
        data = self.client_socket.recv(2)
        vehicle_id = int.from_bytes(data, "big")
        client_sockets[vehicle_id] = self.client_socket
        data = self.client_socket.recv(8)
        while data:
            v_type = int.from_bytes(data[0:2], "big")
            v_id = int.from_bytes(data[2:4], "big")
            x = int.from_bytes(data[4:6], "big")
            y = int.from_bytes(data[6:8], "big")
            location_map[v_id] = (x, y)
            vehicle_types[v_id] = v_type
            data = self.client_socket.recv(8)
        self.client_socket.close()


def server_recv_data(client_socket, client_addr):
    print("Connect data channel from: ", client_addr)
    data = client_socket.recv(2)
    vehicle_id = int.from_bytes(data, "big")
    # if vehicle_id not in client_data_sockets.keys():
    #     client_data_sockets[vehicle_id] = [client_socket]
    # else:
    #     client_data_sockets[vehicle_id].append(client_socket)
    while True:
        data = client_socket.recv(10)
        if len(data) <= 0:
            print("[Helper relay closed]")
            client_socket.close()
            return
        msg_size = int.from_bytes(data[0:4], "big")
        frame_id = int.from_bytes(data[4:6], "big")
        # v_id is the actual pcd captured vehicle, which might be different from sender vehicle id
        v_id = int.from_bytes(data[6:8], "big") 
        data_type = int.from_bytes(data[8:10], "big") 
        # print('recv size ' + str(pcd_size))
        msg = b''
        to_recv = msg_size
        t_recv_start = time.time()
        while len(msg) < msg_size:
            data_recv = client_socket.recv(65536 if to_recv > 65536 else to_recv)
            if len(data_recv) <= 0:
                print("[Helper relay closed]")
                client_socket.close()
                return
            msg += data_recv
            to_recv = msg_size - len(msg)
        assert len(msg) == msg_size
        t_elasped = time.time() - t_recv_start
        if frame_id >= MAX_FRAMES:
            continue
        if data_type == TYPE_PCD:
            print("[Full frame recved] from %d, id %d throughput: %f MB/s" % 
                        (v_id, frame_id, msg_size/1000000.0/t_elasped))
            pcds[v_id][frame_id] = msg
        elif data_type == TYPE_OXTS:
            print("[Oxts recved] from %d, frame id %d" %  (v_id, frame_id))
            oxts[v_id][frame_id] = [float(x) for x in msg.split()]
        
        if len(pcds[v_id][frame_id]) > 0 and len(oxts[v_id][frame_id]) > 0:
            data_ready_matrix[v_id][frame_id] = 1
        
        # print(data_ready_matrix)

def merge_data_when_ready():
    global curr_processed_frame
    while curr_processed_frame < MAX_FRAMES:
        ready = True
        for n in range(curr_vehicles):
            if not data_ready_matrix[n][curr_processed_frame]:
                ready = False
                break
        if ready:
            print("[merge data] merge frame %d ar %f" % (curr_processed_frame, time.time()))
            points_oxts_primary = (np.frombuffer(pcds[0][curr_processed_frame], 
                                    dtype='float32').reshape([-1, 4]), 
                                    oxts[0][curr_processed_frame])
            points_oxts_secondary = []
            for i in range(1,curr_vehicles):
                pcl = np.frombuffer(pcds[i][curr_processed_frame], 
                                dtype='float32').reshape([-1, 4])
                points_oxts_secondary.append((pcl,oxts[i][curr_processed_frame]))
            merged_pcl = pcd_merge.merge(points_oxts_primary, points_oxts_secondary)
            with open('output/merged_%d.bin'%curr_processed_frame, 'w') as f:
                merged_pcl.tofile(f)
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
            self.data_channel_sock.listen(1)
            client_socket, client_address = self.data_channel_sock.accept()
            new_data_recv_thread = threading.Thread(target=server_recv_data, \
                                                    args=(client_socket,client_address))
            new_data_recv_thread.daemon = True
            new_data_recv_thread.start()


def main():
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
        server.listen(1)
        client_socket, client_address = server.accept()
        newthread = ConnectionThread(client_address, client_socket)
        newthread.daemon = True
        newthread.start()


if __name__ == "__main__":
    main()
