# this file handles the main process run by each vehicle node
 # -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stderr = sys.stdout
import threading
import socket
import time
import ptcl.pointcloud
import wwan
import config
import utils
import mobility
import numpy as np
import argparse
import message

HELPEE = 0
HELPER = 1
TYPE_PCD = 0
TYPE_OXTS = 1
FRAMERATE = 5
PCD_ENCODE_LEVEL = 10 # point cloud encode level
PCD_QB = 12 # point cloud quantization bits

sys.stderr = sys.stdout

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', default=0, type=int, help='vehicle id')
parser.add_argument('-d', '--data_path', default='../DeepGTAV-data/object-0227-1/',\
                    type=str, help='point cloud and oxts data path')
parser.add_argument('-l', '--location_file', default="input/object-0227-loc.txt", \
                    type=str, help='location file name')
parser.add_argument('-c', '--helpee_conf', default="input/helpee_conf/helpee-nodes.txt",\
                    type=str, help='helpee nodes configuration file')
parser.add_argument('-f', '--fps', default=1, type=int, help='FPS of pcd data')
args = parser.parse_args()


vehicle_id = args.id
PCD_DATA_PATH = args.data_path + '/velodyne_2/'
OXTS_DATA_PATH = args.data_path + '/oxts/'
LOCATION_FILE = args.location_file
HELPEE_CONF = args.helpee_conf
FRAMERATE = args.fps
print('fps ' + str(FRAMERATE))
connection_state = "Connected"
current_helpee_id = 65535
current_helper_id = 65535
curr_timestamp = 0.0
vehicle_locs = mobility.read_locations(LOCATION_FILE)
self_loc_trace = vehicle_locs[vehicle_id]
self_loc = self_loc_trace[0]
pcd_data_buffer = []
oxts_data_buffer = []


vehicle_seq_dict = {} # store other vehicle's highest seq
control_seq_num = 0 # self seq number

frame_lock = threading.Lock()
v2i_control_socket_lock = threading.Lock()
v2i_control_socket = wwan.setup_p2p_links(vehicle_id, config.server_ip, config.server_ctrl_port)
v2i_data_socket = wwan.setup_p2p_links(vehicle_id, config.server_ip, config.server_data_port)
v2v_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
curr_frame_id = 0
v2v_recved_bytes = 0

helper_data_send_thread = []
helper_data_recv_port = 8080
helper_control_recv_port = 8888
self_ip = "10.0.0." + str(vehicle_id+2)


def throughput_calc_thread():
    """ Thread to calculate V2V thrpt
    """
    global v2v_recved_bytes
    while True:
        print("[relay throughput] %f Mbps %f"%(v2v_recved_bytes*8.0/1000000, time.time()), flush=True)
        v2v_recved_bytes = 0
        time.sleep(1)


def sensor_data_capture(pcd_data_path, oxts_data_path, fps):
    """Thread to capture (read) point cloud file at a certain FPS setting

    Args:
        pcd_data_path (str): path to pcd data
        oxts_data_path (str): path to oxts data
        fps (float): Frame rate to load data to buffer
    """
    for i in range(config.MAX_FRAMES):
        t_s = time.time()
        pcd_f_name = pcd_data_path + "%06d.bin"%i
        raw_pcd = ptcl.pointcloud.read_pointcloud(pcd_f_name)
        pcd, _ = ptcl.pointcloud.dracoEncode(np.frombuffer(raw_pcd, dtype='float32').reshape([-1,4]), 
                                            PCD_ENCODE_LEVEL, PCD_QB)
        pcd_data_buffer.append(pcd)
        oxts_f_name = oxts_data_path + "%06d.txt"%i
        oxts_data_buffer.append(ptcl.pointcloud.read_oxts(oxts_f_name))
        t_elasped = time.time() - t_s
        # print("sleep %f before get the next frame" % (1.0/fps-t_elasped), flush=True)
        # if (1.0/fps-t_elasped) > 0:
        #     time.sleep(1.0/fps-t_elasped)


def send(socket, data, id, type):
    msg_len = len(data)
    header = msg_len.to_bytes(4, "big") + id.to_bytes(2, "big") \
                + vehicle_id.to_bytes(2, "big") + type.to_bytes(2, 'big')
    print("[send header] vehicle %d, frame %d, data len: %d" % (vehicle_id, id, msg_len))
    hender_sent = 0
    while hender_sent < len(header):
        bytes_sent = socket.send(header[hender_sent:])
        hender_sent += bytes_sent
    assert hender_sent == len(header)
    total_sent = 0
    while total_sent < msg_len:
        try:
            bytes_sent = socket.send(data[total_sent:])
            # TODO: check how much bytes sent each time exactly
            print("[Sedning Data] Sent %d bytes" % bytes_sent)
            total_sent += bytes_sent
            if bytes_sent == 0:
                raise RuntimeError("socket connection broken")
        except:
            print('[Send error]')
            # socket.close()
            # return


def v2i_data_send_thread():
    """Thread to handle V2I data sending
    """
    global curr_frame_id
    while connection_state == "Connected":
        frame_lock.acquire()
        if curr_frame_id < config.MAX_FRAMES and curr_frame_id < len(pcd_data_buffer) \
            and curr_frame_id < len(oxts_data_buffer):
            curr_f_id = curr_frame_id
            pcd = pcd_data_buffer[curr_f_id]
            oxts = oxts_data_buffer[curr_f_id]
            curr_frame_id += 1
            frame_lock.release()
            # pcd, _ = ptcl.pointcloud.dracoEncode(np.frombuffer(pcd, dtype='float32').reshape([-1,4]), 
            #                                 PCD_ENCODE_LEVEL, PCD_QB)
            # TODO: maybe compress the frames beforehand, change encode to another thread
            print("[V2I send pcd frame] " + str(curr_f_id) + ' ' + str(time.time()), flush=True)
            send(v2i_data_socket, pcd, curr_f_id, TYPE_PCD)
            send(v2i_data_socket, oxts, curr_f_id, TYPE_OXTS)
            print("[Frame sent finished] " + str(curr_f_id) + ' ' + str(time.time()), flush=True)
        elif curr_frame_id >= config.MAX_FRAMES:
            curr_frame_id = 0
            print('[Max frame reached] finished ' + str(time.time()))
            frame_lock.release()
            # return
        else:
            frame_lock.release()


def self_loc_update_thread():
    """Thread to update self location every 100ms
    """
    global self_loc
    for loc in self_loc_trace:
        self_loc = loc
        time.sleep(0.1)


def is_helper_recv():
    """Check whether current node is helper

    Returns:
        Bool: True for helper, False for helpee
    """
    global connection_state
    return connection_state == "Connected"


def if_recv_nothing_from_server(recv_id):
    """Check if nothing is received from the server (receive id to be a large number)

    Args:
        recv_id (int): receive id of helpee

    Returns:
        Bool: True for nothing received, False else
    """
    return recv_id == 65534


def if_not_assigned_as_helper(recv_id):
    """Check if the response from server is that I'm not assigned as helper

    Args:
        recv_id (int): helpee id to help, if 65535, help no one

    Returns:
        Bool: True for not assigned as helper, False else
    """
    return recv_id == 65535


def v2v_data_recv_thread():
    """ a seperate thread to recv point cloud from helpee node and forward it to the server
    """
    host_ip = ''
    host_port = helper_data_recv_port
    v2v_data_recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    v2v_data_recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    v2v_data_recv_sock.bind((host_ip, host_port))
    while True:
        v2v_data_recv_sock.listen()
        client_socket, client_address = v2v_data_recv_sock.accept()
        print("[get helpee connection] " + str(time.time()))
        new_data_recv_thread = VehicleDataRecvThread(client_socket, client_address)
        new_data_recv_thread.daemon = True
        new_data_recv_thread.start()


def notify_helpee_node(helpee_id):
    """Notify helpee node that I'm the helper assigned to help relay data. Must be sent from a 
    helper

    Args:
        helpee_id (int): helpee id to notify
    """
    # print("[notifying the helpee]")
    send_note_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    msg = vehicle_id.to_bytes(2, 'big')
    header = message.construct_control_msg_header(msg, message.TYPE_ASSIGNMENT)
    helpee_addr = "10.0.0." + str(helpee_id+2)
    message.send_msg(send_note_sock, header, msg, is_udp=True,\
                        remote_addr=(helpee_addr, helper_control_recv_port))
    # send_note_sock.sendto(msg, (helpee_addr, helper_control_recv_port))


class ServerControlThread(threading.Thread):
    """Thread that handle control messages between node and server
    """

    def __init__(self):
        threading.Thread.__init__(self)
        # print("Setup V2I socket connection")

    
    def run(self):
        while True:
            # always receive assignment information from server
            # Note: sending location information is done in VehicleConnThread.run()
            # triggered by receiving location info from helpees
            global current_helpee_id
            helpee_id = wwan.recv_assignment(v2i_control_socket)
            if is_helper_recv():
                if if_recv_nothing_from_server(helpee_id):
                    print("Recv nothing from server " + str(time.time()))
                elif if_not_assigned_as_helper(helpee_id):
                    print("Not assigned as helper.. " + str(time.time()))
                else:
                    if current_helpee_id == helpee_id:
                        # print("already helping node " + str(current_helpee_id))
                        pass
                    else:
                        print("[Helper get assignment from server] helpee_id: " +\
                                 str(helpee_id) + ' ' + str(time.time()))
                        print(self_loc)
                        current_helpee_id = helpee_id
                        notify_helpee_node(helpee_id)
            time.sleep(0.2)


def parse_location_packet_data(data):
    """Parse location packet, packet should be of length 6

    Args:
        data (bytes): raw network packet data to parse

    Returns:
        helpee_id: helpee node id that sends the packet
        [x, y]: location of the helpee node
    """
    # return helpee id, location
    helpee_id = int.from_bytes(data[0:2], "big")
    x = int.from_bytes(data[2:4], "big")
    y = int.from_bytes(data[4:6], "big")
    seq_num = int.from_bytes(data[6:10], "big")
    return helpee_id, [x, y], seq_num


def is_packet_assignment(data):
    """Check packet to be an assignment packet (sent from server to assign helpee node)

    Args:
        data (bytes): raw network packet data to parse

    Returns:
        Bool: True for assignment packets, False else
    """
    return len(data) == 2


class VehicleControlThread(threading.Thread):
    """Thread that handle control messages between nodes (vehicles)
    """

    def __init__(self):
        threading.Thread.__init__(self)
        print("Setup V2V socket to recv broadcast message")
        global v2v_control_socket
        host_ip = ''
        host_port = helper_control_recv_port
        # Use UDP socket for broadcasting
        v2v_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, \
                                                     socket.IPPROTO_UDP)
        v2v_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        v2v_control_socket.bind((host_ip, host_port))
        

    def run(self):
        global current_helper_id, control_seq_num
        while True:
            data, addr = message.recv_msg(v2v_control_socket, message.TYPE_CONTROL_MSG,\
                                            is_udp=True)
            
            msg_size, msg_type = message.parse_control_msg_header(data)
            if connection_state == "Connected":
                if msg_type == message.TYPE_LOCATION:
                    # helper
                    print("[helper recv broadcast] " + str(time.time()))
                    helpee_id, helpee_loc, seq_num = parse_location_packet_data(data[-msg_size:])
                    # send helpee location
                    print((helpee_id, helpee_loc))
                    if helpee_id not in vehicle_seq_dict.keys() \
                        or seq_num > vehicle_seq_dict[helpee_id]:
                        # only send location if received seq num is larger
                        vehicle_seq_dict[helpee_id] = seq_num
                        v2i_control_socket_lock.acquire()
                        wwan.send_location(HELPEE, helpee_id, helpee_loc, v2i_control_socket, seq_num)
                        # send self location
                        # wwan.send_location(HELPER, vehicle_id, self_loc, v2i_control_socket, \
                        #                      control_seq_num) 
                        control_seq_num += 1
                        v2i_control_socket_lock.release()
            elif connection_state == "Disconnected":
                if msg_type == message.TYPE_ASSIGNMENT:
                    helper_id = int.from_bytes(data[-msg_size:], 'big')
                    print("[Helpee get helper assignment] helper_id: "\
                         + str(helper_id) + ' ' + str(time.time()), flush=True)
                    helper_ip = "10.0.0." + str(helper_id+2)   
                    current_helper_id = helper_id
                    new_send_thread = VehicleDataSendThread(helper_ip, helper_data_recv_port)
                    new_send_thread.daemon = True
                    new_send_thread.start()
                    if len(helper_data_send_thread) != 0:
                        helper_data_send_thread[-1].stop()
                    helper_data_send_thread.append(new_send_thread)
                elif msg_type == message.TYPE_LOCATION\
                        and self_ip != addr[0]:
                    helpee_id, helpee_loc, seq_num = parse_location_packet_data(data[-msg_size:])
                    if helpee_id != vehicle_id and (helpee_id not in vehicle_seq_dict.keys() or \
                            seq_num > vehicle_seq_dict[helpee_id]):
                        # helpee only rebroadcast loc not equal to themselves and with larger seq
                        vehicle_seq_dict[helpee_id] = seq_num
                        message.send_msg(v2v_control_socket, data[:message.CONTROL_MSG_HEADER_LEN],\
                                        data[message.CONTROL_MSG_HEADER_LEN:], is_udp=True, \
                                        remote_addr=("10.255.255.255", helper_control_recv_port))


class VehicleDataRecvThread(threading.Thread):
    """Thread that handle data receiving between nodes (vehicles)
        Recv using tcp socket from helpee nodes
    """

    def __init__(self, helpee_socket, helpee_addr):
        threading.Thread.__init__(self)
        self.client_socket = helpee_socket
        self.client_address = helpee_addr
        self._is_closed = False
    

    def run(self):
        global v2v_recved_bytes
        helper_relay_server_sock = wwan.setup_p2p_links(vehicle_id, config.server_ip, 
                                                            config.server_data_port)
        while True and not self._is_closed:
            header_len = 10
            data = b''
            header_to_recv = header_len
            while len(data) < header_len:
                data_recv = self.client_socket.recv(header_to_recv)
                data += data_recv
                if len(data_recv) <= 0:
                    print("[Helpee closed]")
                    self._is_closed = True
                    helper_relay_server_sock.close()
                    return
                header_to_recv -= len(data_recv)
                v2v_recved_bytes += len(data_recv)
            msg_len = int.from_bytes(data[0:4], "big")
            frame_id = int.from_bytes(data[4:6], "big")
            v_id = int.from_bytes(data[6:8], "big")
            type = int.from_bytes(data[8:10], "big")
            assert len(data) == 10
            to_send = header_len
            sent = 0
            while sent < to_send:
                sent_len = helper_relay_server_sock.send(data[sent:])
                sent += sent_len
            to_recv = msg_len
            curr_recv_bytes = 0
            t_start = time.time()
            while curr_recv_bytes < msg_len:
                data = self.client_socket.recv(65536 if to_recv > 65536 else to_recv)
                if len(data) <= 0:
                    print("[Helpee closed]")
                    self._is_closed = True
                    helper_relay_server_sock.close()
                    break
                curr_recv_bytes += len(data)
                to_recv -= len(data)
                v2v_recved_bytes += len(data)
                helper_relay_server_sock.send(data)
            t_elasped = time.time() - t_start
            est_v2v_thrpt = msg_len/t_elasped/1000000.0
            if self._is_closed:
                return
            if type == TYPE_PCD:
                print("[Received a full frame/oxts] %f frame: %d vehicle: %d %f" 
                        % (est_v2v_thrpt, frame_id, v_id, time.time()), flush=True)


class VehicleDataSendThread(threading.Thread):
    """ Thread that handle data sending between nodes (vehicles)
        Used when helpee get assignment and connect to the helper node
    """

    def __init__(self, helper_ip, helper_port):
        threading.Thread.__init__(self)
        self.v2v_data_send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.v2v_data_send_sock.connect((helper_ip, helper_port))
        self.is_helper_alive = True
    
    
    def run(self):
        global curr_frame_id
        while self.is_helper_alive:
            frame_lock.acquire()
            if curr_frame_id < config.MAX_FRAMES and curr_frame_id < len(pcd_data_buffer) \
                and curr_frame_id < len(oxts_data_buffer):
                curr_f_id = curr_frame_id
                pcd = pcd_data_buffer[curr_frame_id]
                oxts = oxts_data_buffer[curr_frame_id]
                curr_frame_id += 1
                frame_lock.release()
                # pcd, _ = ptcl.pointcloud.dracoEncode(np.frombuffer(pcd, dtype='float32').reshape([-1,4]),
                #                                 PCD_ENCODE_LEVEL, PCD_QB)
                print("[V2V send pcd frame] Start sending frame " + str(curr_f_id) + " to helper " \
                         + str(current_helper_id) + ' ' + str(time.time()), flush=True)
                send(self.v2v_data_send_sock, pcd, curr_f_id, TYPE_PCD)
                send(self.v2v_data_send_sock, oxts, curr_f_id, TYPE_OXTS)
            elif curr_frame_id >= config.MAX_FRAMES:
                curr_frame_id = 0
                frame_lock.release()
                print('[Max frame reached] finished ' + str(time.time()))
                # break
            else:
                frame_lock.release()
        print("[Change helper] close the prev conn thread " + str(time.time()))
        self.v2v_data_send_sock.close()


    def stop(self):
        self.is_helper_alive = False


def check_if_disconnected(disconnect_timestamps):
    """ Check whether the vehicle is disconnected to the server

    Args:
        disconnect_timestamps (dictionary): the disconnected timestamp dictionary read from
        LTE trace e.g. dict = {1: [2.1]} means node 1 disconnects at 2.1 sec

    Returns:
        Bool: True for disconnected, False else
    """
    # print("check if V2I conn is disconnected")
    if vehicle_id in disconnect_timestamps.keys():
        if time.time() - curr_timestamp > disconnect_timestamps[vehicle_id][0]:
            return True
        else:
            return False
    else:
        return False


def check_connection_state(disconnect_timestamps):
    """Main thread function to constantly check connection to the server, and broadcast its 
    location if disconnect

    Args:
        disconnect_timestamps (dictionary): the disconnected timestamp dictionary read from
        LTE trace e.g. dict = {1: [2.1]} means node 1 disconnects at 2.1 sec
    """
    global connection_state, control_seq_num
    while True:
        t_start = time.time()
        if connection_state == "Connected":
            # print("Connected to server...")
            if vehicle_id not in disconnect_timestamps.keys():
                v2i_control_socket_lock.acquire()
                wwan.send_location(HELPER, vehicle_id, self_loc, v2i_control_socket,\
                                    control_seq_num) 
                control_seq_num += 1
                v2i_control_socket_lock.release()
        elif connection_state == "Disconnected":
            # print("Disconnected to server... broadcast " + str(time.time()))
            v2i_control_socket_lock.acquire()
            mobility.broadcast_location(vehicle_id, self_loc, v2v_control_socket, control_seq_num)
            control_seq_num += 1
            v2i_control_socket_lock.release()
        if check_if_disconnected(disconnect_timestamps):
            connection_state = "Disconnected"
        t_elasped = time.time() - t_start
        if 0.2-t_elasped > 0:
            time.sleep(0.2-t_elasped)



def main():
    global curr_timestamp
    trace_files = 'trace.txt'
    lte_traces = utils.read_traces(trace_files)
    disconnect_timestamps = utils.process_traces(lte_traces, HELPEE_CONF)
    print("disconnect timestamp")
    print(disconnect_timestamps)
    t_start = time.time()
    sensor_data_capture(PCD_DATA_PATH, OXTS_DATA_PATH, FRAMERATE)
    t_elapsed = time.time() - t_start
    print("read and encode takes %f" % t_elapsed)
    curr_timestamp = time.time()
    v2i_control_thread = ServerControlThread()
    v2i_control_thread.start()
    v2v_control_thread = VehicleControlThread()
    v2v_control_thread.start()
    v2v_data_thread = threading.Thread(target=v2v_data_recv_thread, args=())
    v2v_data_thread.start()
    v2i_data_thread = threading.Thread(target=v2i_data_send_thread, args=())
    v2i_data_thread.start()
    loction_update_thread = threading.Thread(target=self_loc_update_thread, args=())
    loction_update_thread.daemon = True
    loction_update_thread.start()

    # senser_data_capture_thread = threading.Thread(target=sensor_data_capture, \
    #          args=(PCD_DATA_PATH, OXTS_DATA_PATH, FRAMERATE))
    # senser_data_capture_thread.start()

    throughput_thread = threading.Thread(target=throughput_calc_thread, args=())
    throughput_thread.start()

    check_connection_state(disconnect_timestamps)


if __name__ == '__main__':
    main()