# Network setup and data transfer over the Internet paths

import socket
import threading
import time
import utils
import fcntl, os
import network.message

def setup_lte():
    # TODO
    pass


def setup_p2p_links(vehicle_id, ip, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
    client_socket.connect((ip, port))
    # fcntl.fcntl(client_socket, fcntl.F_SETFL, os.O_NONBLOCK)
    # client_socket.setblocking(0)
    # client_socket.settimeout(1)
    msg = vehicle_id.to_bytes(2, 'big')
    client_socket.send(msg)
    return client_socket


def send_location(vehicle_type, vehicle_id, position, client_socket, seq_num):
    v_type = vehicle_type.to_bytes(2, 'big')
    v_id = vehicle_id.to_bytes(2, 'big')
    x = int(position[0]).to_bytes(2, 'big')
    y = int(position[1]).to_bytes(2, 'big')
    seq = seq_num.to_bytes(4, 'big')
    msg = v_type + v_id + x + y + seq
    header = network.message.construct_control_msg_header(msg, network.message.TYPE_LOCATION)
    network.message.send_msg(client_socket, header, msg)


def send_route(vehicle_type, vehicle_id, route_bytes, client_socket, seq_num):
    v_type = vehicle_type.to_bytes(2, 'big')
    v_id = vehicle_id.to_bytes(2, 'big')
    seq = seq_num.to_bytes(4, 'big')
    msg = v_type + v_id + route_bytes + seq
    header = network.message.construct_control_msg_header(msg, network.message.TYPE_ROUTE)
    network.message.send_msg(client_socket, header, msg)


def recv_assignment(client_socket):
    try:
        msg = client_socket.recv(2)
        helpee_id = int.from_bytes(msg, "big")
        return helpee_id
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
