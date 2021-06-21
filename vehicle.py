# this file handles the main process run by each vehicle node
 # -*- coding: utf-8 -*-
import os, sys
import threading
import socket
import time
import pointcloud
import wwan
import wlan
import config
import utils
import mobility

HELPEE = 0
HELPER = 1
vehicle_id = int(sys.argv[1])
connection_state = "Connected"
current_helpee_id = 65535
curr_timestamp = 0.0
vehicle_locs = mobility.read_locations("input/object-0227.txt")
self_loc_trace = vehicle_locs[vehicle_id]
self_loc = self_loc_trace[0]
pcd_data_buffer = []
pcd_data_buffer.append(pointcloud.read_pointcloud('input/single-pointcloud-frame.bin'))
v2i_control_socket = wwan.setup_p2p_links(vehicle_id, config.server_ip, config.server_ctrl_port)
v2i_data_socket = wwan.setup_p2p_links(vehicle_id, config.server_ip, config.server_data_port)
v2v_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

helper_data_send_thread = []
helper_data_recv_port = 8080
helper_control_recv_port = 8888
self_ip = "10.0.0." + str(vehicle_id+2)


def v2i_send_pcd(pcd_data):
    msg_len = len(pcd_data)
    v2i_data_socket.send(msg_len.to_bytes(4, "big"))
    print("send size " + str(msg_len))
    start_t = time.time()
    v2i_data_socket.send(pcd_data)
    end_t = time.time()
    est_v2i_thrpt = msg_len/1000000/(end_t-start_t)



def v2i_data_send_thread():
    while connection_state == "Connected" and len(pcd_data_buffer) != 0:
        # v2i_send_pcd(pcd_data_buffer[0])
        msg_len = len(pcd_data_buffer[0])
        v2i_data_socket.send(msg_len.to_bytes(4, "big"))
        total = 0
        while total < msg_len:
            try:
                bytes_sent = v2i_data_socket.send(pcd_data_buffer[0][total:])
                if bytes_sent == 0:
                    raise RuntimeError("socket connection broken")
            except:
                print("V2I link disconnected")
                break
            total = total + bytes_sent
        print("[V2I send pcd data] From original vehicle " + str(time.time()))
        # time.sleep(0.1)


def self_loc_update_thread():
    global self_loc
    for loc in self_loc_trace:
        self_loc = loc
        time.sleep(0.1)


def is_helper_recv():
    global connection_state
    print(connection_state)
    return connection_state == "Connected"


def if_recv_nothing_from_server(recv_id):
    return recv_id == 65534


def if_not_assigned_as_helper(recv_id):
    return recv_id == 65535


def setup_data_recv_thread():
    # a seperate thread to recv point cloud and forward it to the server
    host_ip = ''
    host_port = 8080
    v2v_data_recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    v2v_data_recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    v2v_data_recv_sock.bind((host_ip, host_port))
    v2v_data_recv_sock.listen(1)
    while True:
        v2v_data_recv_sock.listen(1)
        client_socket, client_address = v2v_data_recv_sock.accept()
        print("[get helpee connection] " + str(time.time()))
        new_data_recv_thread = VehicleDataRecvThread(client_socket, client_address)
        new_data_recv_thread.daemon = True
        new_data_recv_thread.start()


def setup_data_send_socket(helper_id):
    pass


def close_old_data_recv_socket():
    # close the old recv socket for the previous helpee
    # need to confirm the data has been sent and then close the socket
    pass


def notify_helpee_node(helpee_id):
    print("notifying the helpee node to send data")
    send_note_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    msg = vehicle_id.to_bytes(2, 'big')
    helpee_addr = "10.0.0." + str(helpee_id+2)
    send_note_sock.sendto(msg, (helpee_addr, helper_control_recv_port))


class ServerControlThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        print("Setup V2I socket connection")

    
    def run(self):
        while True:
            # always receive assignment information from server
            # Note: sending location information is done in VehicleConnThread.run()
            # triggered by receiving location info from helpees
            global current_helpee_id
            helpee_id = wwan.recv_assignment(v2i_control_socket)
            if is_helper_recv():
                if if_recv_nothing_from_server(helpee_id):
                    print("Recv nothing from server")
                elif if_not_assigned_as_helper(helpee_id):
                    print("Not assigned as helper..")
                else:
                    if current_helpee_id == helpee_id:
                        print("already helping this node")
                    else:
                        print("[Get assignment] " + str(helpee_id) + ' ' + str(time.time()))
                        print(self_loc)
                        current_helpee_id = helpee_id
                        notify_helpee_node(helpee_id)
                        # setup_data_recv_socket()
                        # close_old_data_recv_socket() this is done in recv thread
            time.sleep(0.2)


def parse_location_packet_data(data):
    # return helpee id, location
    helpee_id = int.from_bytes(data[0:2], "big")
    x = int.from_bytes(data[2:4], "big")
    y = int.from_bytes(data[4:6], "big")
    return helpee_id, [x, y]


def is_packet_assignment(data):
    return len(data) == 2


class VehicleControlThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        print("Setup V2V socket to recv broadcast message")
        global v2v_control_socket
        host_ip = ''
        host_port = 8888
        # Use UDP socket for broadcasting
        v2v_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, \
                                                     socket.IPPROTO_UDP)
        v2v_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        v2v_control_socket.bind((host_ip, host_port))
        

    def run(self):
        while True:
            data, addr = v2v_control_socket.recvfrom(1024)            
            if connection_state == "Connected":
                # helper
                print("helper recv broadcast loc " + str(time.time()))
                # print("send self loc, helpee loc to the server")
                helpee_id, helpee_loc = parse_location_packet_data(data)
                # send helpee location
                wwan.send_location(HELPEE, helpee_id, helpee_loc, v2i_control_socket)
                # send self location
                wwan.send_location(HELPER, vehicle_id, self_loc, v2i_control_socket) 
            else:
                if is_packet_assignment(data):
                    print("[helper assignment] " + str(time.time()))
                    helper_id = int.from_bytes(data, 'big')
                    helper_ip = "10.0.0." + str(helper_id+2)                    
                    new_send_thread = VehicleDataSendThread(helper_ip, helper_data_recv_port)
                    new_send_thread.daemon = True
                    new_send_thread.start()
                    helper_data_send_thread.append(new_send_thread)
                elif self_ip != addr[0]:
                    # helpee
                    # print("helpee recv broadcast loc from others, do flooding")
                    v2v_control_socket.sendto(data, ("10.255.255.255", 8888))


class VehicleDataRecvThread(threading.Thread):

    def __init__(self, helpee_socket, helpee_addr):
        threading.Thread.__init__(self)
        self.client_socket = helpee_socket
        self.client_address = helpee_addr
        self._is_closed = False
    

    def run(self):
        helper_relay_server_sock = wwan.setup_p2p_links(vehicle_id, config.server_ip, 
                                                            config.server_data_port)
        while True:
            data = self.client_socket.recv(4)
            msg_len = int.from_bytes(data, "big")
            helper_relay_server_sock.send(data)
            to_recv = msg_len
            curr_recv_bytes = 0
            t_start = time.time()
            while curr_recv_bytes < msg_len:
                data = self.client_socket.recv(65536 if to_recv > 65536 else to_recv)
                curr_recv_bytes += len(data)
                to_recv -= len(data)
                helper_relay_server_sock.send(data)
                if data == -1:
                    print("[Helpee closed]")
                    self._is_closed = True
                    helper_relay_server_sock.close()
                    break
            t_elasped = time.time() - t_start
            est_v2v_thrpt = msg_len/t_elasped/1000000
            print("[Received a full frame] %f %f" % (est_v2v_thrpt, time.time()))
            if self._is_closed:
                break


class VehicleDataSendThread(threading.Thread):

    def __init__(self, helper_ip, helper_port):
        threading.Thread.__init__(self)
        self.v2v_data_send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.v2v_data_send_sock.connect((helper_ip, helper_port))
        self.is_helper_alive = True
    
    
    def run(self):
        while self.is_helper_alive:
            if len(pcd_data_buffer) != 0:
                print("[Sending data] Start sending a frame to helper " + str(time.time()))
                msg_len = len(pcd_data_buffer[0])
                self.v2v_data_send_sock.send(msg_len.to_bytes(4, "big"))
                sent_total = 0
                while sent_total < msg_len:
                    bytes_sent = self.v2v_data_send_sock.send(pcd_data_buffer[0][sent_total:])
                    sent_total += bytes_sent
                # pcd_data_buffer = pcd_data_buffer[1:]
            

def check_if_disconnected(disconnect_timestamps):
    # print("check if V2I conn is disconnected")
    if vehicle_id in disconnect_timestamps.keys():
        if time.time() - curr_timestamp > disconnect_timestamps[vehicle_id][0]:
            return True
        else:
            return False
    else:
        return False


def check_connection_state(disconnect_timestamps):
    global connection_state
    while True:
        if connection_state == "Connected":
            # print("Connected to server...")
            pass
        elif connection_state == "Disconnected":
            # TODO: setup a timer to resend location information every x ms
            print("Disconnected to server... broadcast " + str(time.time()))
            mobility.broadcast_location(vehicle_id, self_loc, v2v_control_socket)
        if check_if_disconnected(disconnect_timestamps):
            connection_state = "Disconnected"
        time.sleep(0.2)


# TODO: add a pcd data capture thread to read point cloud data in to pcd_data_buffer periodically

def main():
    trace_files = 'trace.txt'
    lte_traces = utils.read_traces(trace_files)
    disconnect_timestamps = utils.process_traces(lte_traces)
    curr_timestamp = time.time()
    v2i_thread = ServerControlThread()
    v2i_thread.start()
    v2v_control_thread = VehicleControlThread()
    v2v_control_thread.start()
    v2v_data_thread = threading.Thread(target=setup_data_recv_thread, args=())
    v2v_data_thread.start()
    v2i_data_thread = threading.Thread(target=v2i_data_send_thread, args=())
    v2i_data_thread.start()
    loction_update_thread = threading.Thread(target=self_loc_update_thread, args=())
    loction_update_thread.daemon = True
    loction_update_thread.start()

    check_connection_state(disconnect_timestamps)


if __name__ == '__main__':
    main()