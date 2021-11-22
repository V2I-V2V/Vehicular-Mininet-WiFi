# Network setup and data transfer over the Internet paths

import socket
import threading
import time
import utils
import fcntl, os
import network.message
import config
import mobility_noise
import pickle


def setup_p2p_links(vehicle_id, ip, port, recv_sched_scheme=False):
    print("connect to", ip, port)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
    client_socket.connect((ip, port))
    # client_socket.setblocking(0)
    # client_socket.settimeout(1)
    msg = vehicle_id.to_bytes(2, 'big')
    client_socket.send(msg)
    if recv_sched_scheme:
        encoded_sched = client_socket.recv(2)
        sched_mode_int = int.from_bytes(encoded_sched, 'big')
        scheduler_mode = config.map_int_encoding_to_scheduler[sched_mode_int]
        return client_socket, scheduler_mode
    else:
        return client_socket


def send_location(vehicle_type, vehicle_id, position, client_socket, seq_num, add_noise=True):
    v_type = vehicle_type.to_bytes(2, 'big')
    v_id = vehicle_id.to_bytes(2, 'big')
    if add_noise:
        loc_x, loc_y = mobility_noise.add_random_noise_on_loc(position[0], position[1], std_deviation=30.0)
    else:
        loc_x, loc_y = position[0], position[1]
    x = int(loc_x).to_bytes(2, 'big')
    y = int(loc_y).to_bytes(2, 'big')
    seq = seq_num.to_bytes(4, 'big')
    msg = v_type + v_id + x + y + seq
    header = network.message.construct_control_msg_header(msg, network.message.TYPE_LOCATION)
    print('[Loc msg size] ', len(header)+len(msg), time.time())
    network.message.send_msg(client_socket, header, msg)


def send_route(vehicle_type, vehicle_id, route_bytes, client_socket, seq_num):
    v_type = vehicle_type.to_bytes(2, 'big')
    v_id = vehicle_id.to_bytes(2, 'big')
    seq = seq_num.to_bytes(4, 'big')
    msg = v_type + v_id + route_bytes + seq
    header = network.message.construct_control_msg_header(msg, network.message.TYPE_ROUTE)
    print('[route msg] %d %f'%(len(header)+len(msg), time.time()))
    network.message.send_msg(client_socket, header, msg)


def recv_control_msg(client_socket):
    try:
        header, payload = network.message.recv_msg(client_socket, network.message.TYPE_CONTROL_MSG)
        _, msg_type = network.message.parse_control_msg_header(header)
        if msg_type == network.message.TYPE_ASSIGNMENT:
            # msg = client_socket.recv(2)
            helpee_id = int.from_bytes(payload, "big")
            return helpee_id, msg_type
        elif msg_type == network.message.TYPE_GROUP:
            group_id = pickle.loads(payload)
            return group_id, msg_type
        elif msg_type == network.message.TYPE_FALLBACK:
            return 0, msg_type
        elif msg_type == network.message.TYPE_RECONNECT:
            return 0, msg_type
    except socket.timeout:
        # print("got error during recv")
        return 65534


class ClientThread(threading.Thread):

    def __init__(self, vehicle_id):
        threading.Thread.__init__(self)
        self.vehicle_id = vehicle_id


    def run(self):
        self.client_socket = setup_p2p_links(self.vehicle_id, "127.0.0.1", 6666)
        for j in range(10):
            send_location(0, 0, [utils.random_int(0, 400), utils.random_int(0, 400)], self.client_socket)
            time.sleep(1)


def main():
    for i in range(8):
        client_thread = ClientThread(i)
        client_thread.daemon = True
        client_thread.start()

if __name__ == "__main__":
    main()
