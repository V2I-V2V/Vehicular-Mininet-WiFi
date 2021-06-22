# Vehicular perception server

import scheduling
import socket
import threading
import time
import pcd_merge
import numpy as np

location_map = {}
client_sockets = {}
client_data_sockets = {}
vehicle_types = {} # 0 for helpee, 1 for helper
vehicle_pcds = {}
helper_helpee_socket_map = {}
current_assignment = {}

oxts_path1 = '/home/ry4nzzz/DeepGTAV-data/object-0227-1/oxts/'
oxts_path2 = '/home/ry4nzzz/DeepGTAV-data/object-0227-1/alt_perspective/0022786/oxts/'
oxts_path3 = '/home/ry4nzzz/DeepGTAV-data/object-0227-1/alt_perspective/0037122/oxts/'
oxts_path4 = '/home/ry4nzzz/DeepGTAV-data/object-0227-1/alt_perspective/0191023/oxts/'
oxts_path5 = '/home/ry4nzzz/DeepGTAV-data/object-0227-1/alt_perspective/0399881/oxts/'
oxts_path6 = '/home/ry4nzzz/DeepGTAV-data/object-0227-1/alt_perspective/0735239/oxts/'
f = open(oxts_path1+'000000.txt', 'r')
oxts1 = [float(x) for x in f.read().split()]
f.close()
f = open(oxts_path2+'000000.txt', 'r')
oxts2 = [float(x) for x in f.read().split()]
f.close()
f = open(oxts_path3+'000000.txt', 'r')
oxts3 = [float(x) for x in f.read().split()]
f.close()
f = open(oxts_path4+'000000.txt', 'r')
oxts4 = [float(x) for x in f.read().split()]
f.close()
f = open(oxts_path5+'000000.txt', 'r')
oxts5 = [float(x) for x in f.read().split()]
f.close()
f = open(oxts_path6+'000000.txt', 'r')
oxts6 = [float(x) for x in f.read().split()]
f.close()


class SchedThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)


    def run(self):
        global current_assignment
        while True:
            print(time.time())
            print(location_map)
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


def server_recv_pcd_data(client_socket, client_addr):
    print("Connect data channel from: ", client_addr)
    data = client_socket.recv(2)
    vehicle_id = int.from_bytes(data, "big")
    if vehicle_id not in client_data_sockets.keys():
        client_data_sockets[vehicle_id] = [client_socket]
    else:
        client_data_sockets[vehicle_id].append(client_socket)
    while True:
        data = client_socket.recv(4)
        if len(data) <= 0:
            print("[Helper relay closed]")
            break
        pcd_size = int.from_bytes(data, "big")
        # print('recv size ' + str(pcd_size))
        msg = b''
        to_recv = pcd_size
        t_recv_start = time.time()
        while len(msg) < pcd_size:
            data_recv = client_socket.recv(65536 if to_recv > 65536 else to_recv)
            if len(data_recv) <= 0:
                print("[Helper relay closed]")
                return
            msg += data_recv
            to_recv = pcd_size - len(msg)
            # to_recv -= len(data_recv)
        assert len(msg) == pcd_size
        t_elasped = time.time() - t_recv_start
        print("[Full frame recved] from %d, throughput: %f" % 
                    (vehicle_id, pcd_size/1000000.0/t_elasped))
        index = client_data_sockets[vehicle_id].index(client_socket)
        if index > 0: # this is a helper relay socket
            real_sender = current_assignment[vehicle_id]
            vehicle_pcds[real_sender] = msg
        else:
            vehicle_pcds[vehicle_id] = msg

        if len(vehicle_pcds.keys()) == 6:
            pcl1 = np.frombuffer(vehicle_pcds[0], dtype='float32').reshape([-1, 4])
            pcl2 = np.frombuffer(vehicle_pcds[1], dtype='float32').reshape([-1, 4])
            pcl3 = np.frombuffer(vehicle_pcds[2], dtype='float32').reshape([-1, 4])
            pcl4 = np.frombuffer(vehicle_pcds[3], dtype='float32').reshape([-1, 4])
            pcl5 = np.frombuffer(vehicle_pcds[4], dtype='float32').reshape([-1, 4])
            pcl6 = np.frombuffer(vehicle_pcds[5], dtype='float32').reshape([-1, 4])
            points_oxts_primary = (pcl1,oxts1)
            points_oxts_secondary = []
            points_oxts_secondary.append((pcl2,oxts2))
            points_oxts_secondary.append((pcl3,oxts3))
            points_oxts_secondary.append((pcl4,oxts4))
            points_oxts_secondary.append((pcl5,oxts5))
            points_oxts_secondary.append((pcl6,oxts6))
            pcl = pcd_merge.merge(points_oxts_primary, points_oxts_secondary)
            with open('merged.bin', 'w') as f:
                pcl.tofile(f)


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
            new_data_recv_thread = threading.Thread(target=server_recv_pcd_data, \
                                                    args=(client_socket,client_address))
            new_data_recv_thread.daemon = True
            new_data_recv_thread.start()


def main():
    HOST = ''
    PORT = 6666
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    print("Vehicular perception server started")
    print("Waiting for client request..")
    sched_thread = SchedThread()
    sched_thread.daemon = True
    sched_thread.start()
    data_channel_thread = DataConnectionThread()
    data_channel_thread.daemon = True
    data_channel_thread.start()
    while True:
        server.listen(1)
        client_socket, client_address = server.accept()
        newthread = ConnectionThread(client_address, client_socket)
        newthread.daemon = True
        newthread.start()


if __name__ == "__main__":
    main()
