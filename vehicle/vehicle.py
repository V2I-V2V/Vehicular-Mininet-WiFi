# this file handles the main process run by each vehicle node
 # -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stderr = sys.stdout
import threading
import socket
import time
import ptcl.pointcloud
import ptcl.partition
import wwan, wlan
import config
import utils
import mobility
import route
import numpy as np
import argparse
import network.message
import bisect

# Define some constants
HELPEE = 0
HELPER = 1
TYPE_PCD = 0
TYPE_OXTS = 1
FRAMERATE = 5
PCD_ENCODE_LEVEL = 10 # point cloud encode level
PCD_QB = 12 # point cloud quantization bits
NO_ADAPTIVE_ENCODE = 0
ADAPTIVE_ENCODE = 1
ADAPTIVE_ENCODE_FULL_CHUNK = 2

sys.stderr = sys.stdout

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', default=0, type=int, help='vehicle id')
parser.add_argument('-d', '--data_path', default='~/DeepGTAV-data/object-0227-1/',\
                    type=str, help='point cloud and oxts data path')
parser.add_argument('--data_type', default="GTA", choices=["GTA", "Carla"])
parser.add_argument('-l', '--location_file', default=os.path.dirname(os.path.abspath(__file__)) + "/input/object-0227-loc.txt", \
                    type=str, help='location file name')
parser.add_argument('-c', '--helpee_conf', default=os.path.dirname(os.path.abspath(__file__)) + "/input/helpee_conf/helpee-nodes.txt",\
                    type=str, help='helpee nodes configuration file')
parser.add_argument('-f', '--fps', default=1, type=int, help='FPS of pcd data')
parser.add_argument('-n', '--disable_control', default=0, type=int, help='disable control msgs')
parser.add_argument('--adaptive', default=0, type=int, \
    help="adaptive encoding type (0 for no adaptive encoding, 1 for adaptive, 2 for adaptive but always use full 4 chunks")
parser.add_argument('--adapt_skip_frames', default=False, action="store_true", \
    help="enable adaptive frame skipping when sending takes too long")
args = parser.parse_args()

control_msg_disabled = True if args.disable_control == 1 else False
vehicle_id = args.id
is_adaptive_frame_skipped = args.adapt_skip_frames

pcd_data_type = args.data_type
if args.data_type == "GTA":
    PCD_DATA_PATH = args.data_path + '/velodyne_2/'
    OXTS_DATA_PATH = args.data_path + '/oxts/'
else:
    PCD_DATA_PATH = args.data_path
    OXTS_DATA_PATH = args.data_path

LOCATION_FILE = args.location_file
HELPEE_CONF = args.helpee_conf
FRAMERATE = args.fps
ADAPTIVE_ENCODE_TYPE = args.adaptive
print('fps ' + str(FRAMERATE))
curr_frame_rate = FRAMERATE
connection_state = "Connected"
current_helpee_id = 65535
current_helper_id = 65535
start_timestamp = 0.0
last_frame_sent_ts = 0.0
vehicle_locs = mobility.read_locations(LOCATION_FILE)
self_loc_trace = vehicle_locs[vehicle_id]
self_loc = self_loc_trace[0]
if len(self_loc_trace) > 5:
    self_loc = self_loc[5:] # manually sync location update with vechiular_perception.py
pcd_data_buffer = []
oxts_data_buffer = []
e2e_frame_latency = {}
e2e_frame_latency_lock = threading.Lock()
frame_sent_time = {}

def self_loc_update_thread():
    """Thread to update self location every 100ms
    """
    global self_loc
    print("[start loc update] at %f" % time.time())
    # loc_log = open('%d_v.txt'%vehicle_id, 'w+')
    for loc in self_loc_trace:
        t_s = time.time()
        self_loc = loc
        # loc_log.write(str(time.time()) + ' ' + str(self_loc[0]) + ' ' +str(self_loc[1]) + '\n')
        t_passed = time.time() - t_s
        if t_passed < 0.1:
            time.sleep(0.1-t_passed)



vehicle_seq_dict = {} # store other vehicle's highest seq
control_seq_num = 0 # self seq number

frame_lock = threading.Lock()
v2i_control_socket_lock = threading.Lock()
v2i_control_socket, scheduler_mode = wwan.setup_p2p_links(vehicle_id, config.server_ip, \
                                        config.server_ctrl_port, recv_sched_scheme=True)
print("server sched mode", scheduler_mode)
v2i_data_socket = wwan.setup_p2p_links(vehicle_id, config.server_ip, config.server_data_port)
v2v_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
curr_frame_id = 0
v2v_recved_bytes = 0

helper_data_send_thread = []
helper_data_recv_port = 8080
helper_control_recv_port = 8888
self_ip = "10.0.0." + str(vehicle_id+2)
capture_finished = False

def throughput_calc_thread():
    """ Thread to calculate V2V thrpt
    """
    global v2v_recved_bytes
    while True:
        print("[relay throughput] %f Mbps %f"%(v2v_recved_bytes*8.0/1000000, time.time()))
        v2v_recved_bytes = 0
        time.sleep(1)


def sensor_data_capture(pcd_data_path, oxts_data_path, fps):
    """Thread to capture (read) point cloud file at a certain FPS setting

    Args:
        pcd_data_path (str): path to pcd data
        oxts_data_path (str): path to oxts data
        fps (float): Frame rate to load data to buffer
    """
    global capture_finished
    for i in range(config.MAX_FRAMES):
        if pcd_data_type == "GTA":
            pcd_f_name = pcd_data_path + "%06d.bin"%i
            oxts_f_name = oxts_data_path + "%06d.txt"%i
        elif pcd_data_type == "Carla":
            pcd_f_name = pcd_data_path + str(1000+i) + ".npy"
            oxts_f_name = oxts_data_path + str(1000+i) + ".trans.npy"
        oxts_data_buffer.append(ptcl.pointcloud.read_oxts(oxts_f_name, pcd_data_type))
        pcd_np = ptcl.pointcloud.read_pointcloud(pcd_f_name, pcd_data_type)

        if ADAPTIVE_ENCODE_TYPE == NO_ADAPTIVE_ENCODE:
            partitioned = ptcl.partition.simple_partition(pcd_np, 20)
            partitioned = pcd_np
            pcd, ratio = ptcl.pointcloud.dracoEncode(partitioned, PCD_ENCODE_LEVEL, PCD_QB)
            pcd_data_buffer.append(pcd)
        else:            
            partitions = ptcl.partition.layered_partition(pcd_np, [5, 8, 15])
            encodeds = []
            for partition in partitions:
                encoded, ratio = ptcl.pointcloud.dracoEncode(partition, PCD_ENCODE_LEVEL, PCD_QB)
                encodeds.append(encoded)
            pcd_data_buffer.append(encodeds)
        # if pcd_data_type == "GTA":
        #     oxts_f_name = oxts_data_path + "%06d.txt"%i
        #     oxts_data_buffer.append(ptcl.pointcloud.read_oxts(oxts_f_name))
        # t_elapsed = time.time() - t_s
        # print("sleep %f before get the next frame" % (1.0/fps-t_elapsed))
        # if (1.0/fps-t_elapsed) > 0:
        #     time.sleep(1.0/fps-t_elapsed)
    capture_finished = True


def get_updated_fps(metric):
    if metric >= 1:
        # Avg latency over 1 sec, min fps adjust to 2
        return 2
    elif metric >= 0.5:
        return 4
    elif metric > 0.1:
        return 7
    elif metric <= 0.1:
        return 10

def get_encoded_frame(frame_id, metric):
    if ADAPTIVE_ENCODE_TYPE == NO_ADAPTIVE_ENCODE:
        cnt = 1
        encoded_frame = pcd_data_buffer[frame_id % config.MAX_FRAMES]
        print("frame id: " + str(frame_id) + " latency: " + str(metric) + " number of chunks: " + str(1))
    elif ADAPTIVE_ENCODE_TYPE == ADAPTIVE_ENCODE:
        encoded_frame = pcd_data_buffer[frame_id % config.MAX_FRAMES][0]
        cnt = 1
        if metric < 0.3:
            encoded_frame += pcd_data_buffer[frame_id % config.MAX_FRAMES][1]
            cnt += 1
        if metric < 0.2:
            encoded_frame += pcd_data_buffer[frame_id % config.MAX_FRAMES][2]
            cnt += 1
        if metric < 0.1:
            encoded_frame += pcd_data_buffer[frame_id % config.MAX_FRAMES][3]
            cnt += 1
        print("frame id: " + str(frame_id) + " latency: " + str(metric) + " number of chunks: " + str(cnt))
    elif ADAPTIVE_ENCODE_TYPE == ADAPTIVE_ENCODE_FULL_CHUNK:
        cnt = 4
        encoded_frame = pcd_data_buffer[frame_id % config.MAX_FRAMES][0] + \
                        pcd_data_buffer[frame_id % config.MAX_FRAMES][1] + \
                        pcd_data_buffer[frame_id % config.MAX_FRAMES][2] + \
                        pcd_data_buffer[frame_id % config.MAX_FRAMES][3]
        print("frame id: " + str(frame_id) + " latency: " + str(metric) + " number of chunks: " + str(4))
    return encoded_frame, cnt
    


def get_latency(e2e_frame_latency):
    recent_latency = 0
    if len(e2e_frame_latency) == 0:
        return 0
    e2e_frame_latency_lock.acquire()
    recent_latencies = sorted(e2e_frame_latency.items(), key=lambda item: -item[0])
    e2e_frame_latency_lock.release()
    cnt = 0
    for id, latency in recent_latencies:
        cnt += 1
        recent_latency += latency
        # print(id, latency)
        if cnt == 5:
            break
    recent_latency /= cnt
    return recent_latency


def send(socket, data, id, type, num_chunks=1, chunks=None):
    msg_len = len(data)
    header = network.message.construct_data_msg_header(data, type, id, vehicle_id, num_chunks, chunks)
    print("[send header] vehicle %d, frame %d, data len: %d" % (vehicle_id, id, msg_len))
    hender_sent = 0
    while hender_sent < len(header):
        try:
            bytes_sent = socket.send(header[hender_sent:])
            hender_sent += bytes_sent
        except:
            print('[Send error]')
            return False
    total_sent = 0
    while total_sent < msg_len:
        try:
            bytes_sent = socket.send(data[total_sent:])
            print("[Sedning Data] Sent %d bytes" % bytes_sent)
            total_sent += bytes_sent
            # if bytes_sent == 0:
            #     raise RuntimeError("socket connection broken")
        except:
            print('[Send error]')
            return False
    return True

def get_frame_ready_timestamp(frame_id, fps):
    return start_timestamp + frame_id * (1/fps)

def get_curr_tranmist_frame_id():
    return int((time.time() - start_timestamp)*FRAMERATE)

def v2i_data_send_thread():
    """Thread to handle V2I data sending
    """
    global curr_frame_id, last_frame_sent_ts, curr_frame_rate
    ack_recv_thread = threading.Thread(target=v2i_ack_recv_thread)
    ack_recv_thread.start()
    while True:
        if connection_state == "Connected":
            t_start = time.time()
            frame_lock.acquire()
            curr_f_id = curr_frame_id if not is_adaptive_frame_skipped else get_curr_tranmist_frame_id()
            data_f_id = curr_f_id % config.MAX_FRAMES
            # pcd = pcd_data_buffer[data_f_id]
            # pcd = pcd_data_buffer[data_f_id][0]
            pcd, num_chunks = get_encoded_frame(curr_frame_id, get_latency(e2e_frame_latency))
            curr_frame_id += 1
            frame_lock.release()
            last_frame_sent_ts = time.time()
            # last_frame_sent_ts should be the timestamp for finishing the sensor data capture
            last_frame_sent_ts = get_frame_ready_timestamp(curr_f_id, FRAMERATE)
            print("[V2I send pcd frame] " + str(curr_f_id) + ' ' + str(last_frame_sent_ts), flush=True)
            frame_sent_time[curr_f_id] = time.time()
            send(v2i_data_socket, pcd, curr_f_id, TYPE_PCD, num_chunks, pcd_data_buffer[data_f_id][:num_chunks])
            # if pcd_data_type == "GTA":
            oxts = oxts_data_buffer[data_f_id]
            send(v2i_data_socket, oxts, curr_f_id, TYPE_OXTS)
            # print("[Frame sent finished] " + str(curr_f_id) + ' ' + str(time.time()-last_frame_sent_ts))
            t_elapsed = time.time() - t_start
            if capture_finished and is_adaptive_frame_skipped:
                curr_frame_rate = get_updated_fps(get_latency(e2e_frame_latency))
                print("Update framerate: ", curr_frame_rate, time.time())
            if capture_finished and (1.0/curr_frame_rate-t_elapsed) > 0:
                print("capture finished, sleep %f" % (1.0/curr_frame_rate-t_elapsed))
                time.sleep(1.0/curr_frame_rate-t_elapsed)
            # if curr_frame_rate != 10:
            #     frame_lock.acquire()
            #     curr_frame_id += int(10/curr_frame_rate)
            #     frame_lock.release()
        else:
            time.sleep(0.1)


def v2i_ack_recv_thread():
    while True:
        ack = v2i_data_socket.recv(2)
        frame_id = int.from_bytes(ack, 'big')
        frame_latency = time.time() - frame_sent_time[frame_id]
        print("[Recv ack from server] frame %d, latency %f"%(frame_id, frame_latency))
        e2e_frame_latency_lock.acquire()
        e2e_frame_latency[frame_id] = frame_latency
        e2e_frame_latency_lock.release()


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
    header = network.message.construct_control_msg_header(msg, network.message.TYPE_ASSIGNMENT)
    helpee_addr = "10.0.0." + str(helpee_id+2)
    network.message.send_msg(send_note_sock, header, msg, is_udp=True,\
                        remote_addr=(helpee_addr, helper_control_recv_port))


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
                    print("[Helper get assignment from server] helpee_id: " +\
                                str(helpee_id) + ' ' + str(time.time()))
                    # print(self_loc)
                    current_helpee_id = helpee_id
                    notify_helpee_node(helpee_id)
            time.sleep(0.2)


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
        v2v_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        v2v_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        v2v_control_socket.bind((host_ip, host_port))
        

    def run(self):
        global current_helper_id, control_seq_num
        while True:
            data, addr = network.message.recv_msg(v2v_control_socket, network.message.TYPE_CONTROL_MSG,\
                                            is_udp=True)
            msg_size, msg_type = network.message.parse_control_msg_header(data)
            if connection_state == "Connected":
                # This vehicle is a helper now
                if msg_type == network.message.TYPE_LOCATION:
                    print("[helper recv location broadcast] " + str(time.time()))
                    helpee_id, helpee_loc, seq_num = network.message.vehicle_parse_location_packet_data(data[-msg_size:])
                    # send helpee location
                    print((helpee_id, helpee_loc))
                    if helpee_id not in vehicle_seq_dict.keys() \
                        or seq_num > vehicle_seq_dict[helpee_id]:
                        # only send location if received seq num is larger
                        vehicle_seq_dict[helpee_id] = seq_num
                        v2i_control_socket_lock.acquire()
                        wwan.send_location(HELPEE, helpee_id, helpee_loc, v2i_control_socket, seq_num)
                        control_seq_num += 1
                        v2i_control_socket_lock.release()
                elif msg_type == network.message.TYPE_ROUTE:
                    print("[helper recv route broadcast] " + str(time.time()))
                    helpee_id, route_bytes, seq_num = network.message.vehicle_parse_route_packet_data(data[-msg_size:])
                    # forward helpee route
                    if helpee_id not in vehicle_seq_dict.keys() \
                        or seq_num > vehicle_seq_dict[helpee_id]:
                        print("[helper forwarding route broadcast] " + str(time.time()))
                        # only send location if received seq num is larger
                        vehicle_seq_dict[helpee_id] = seq_num
                        v2i_control_socket_lock.acquire()
                        wwan.send_route(HELPEE, helpee_id, route_bytes, v2i_control_socket, seq_num)
                        control_seq_num += 1
                        v2i_control_socket_lock.release()
                elif msg_type == network.message.TYPE_SOS and self_ip != addr[0]:
                    # helpee run in distributed way, send back help msg
                    print("[helper recv sos broadcast] " + str(addr) + str(time.time()))
                    wlan.echo_sos_msg(v2v_control_socket, vehicle_id, addr)                
            elif connection_state == "Disconnected":
                # This vehicle is a helpee now
                if msg_type == network.message.TYPE_ASSIGNMENT:
                    helper_id = int.from_bytes(data[-msg_size:], 'big')
                    if helper_id != current_helper_id:
                        print("[Helpee get helper assignment] helper_id: "\
                            + str(helper_id) + ' ' + str(time.time()), flush=True)
                        helper_ip = "10.0.0." + str(helper_id+2)   
                        new_send_thread = VehicleDataSendThread(helper_ip, helper_data_recv_port, helper_id)
                        new_send_thread.daemon = True
                        new_send_thread.start()
                        if len(helper_data_send_thread) != 0:
                            helper_data_send_thread[-1].stop()
                        helper_data_send_thread.append(new_send_thread)
                elif msg_type == network.message.TYPE_LOCATION and self_ip != addr[0]:
                    helpee_id, helpee_loc, seq_num = network.message.vehicle_parse_location_packet_data(data[-msg_size:])
                    if helpee_id != vehicle_id and (helpee_id not in vehicle_seq_dict.keys() or \
                            seq_num > vehicle_seq_dict[helpee_id]):
                        # helpee only rebroadcast loc not equal to themselves and with larger seq
                        vehicle_seq_dict[helpee_id] = seq_num
                        network.message.send_msg(v2v_control_socket, data[:network.message.CONTROL_MSG_HEADER_LEN],\
                                        data[network.message.CONTROL_MSG_HEADER_LEN:], is_udp=True, \
                                        remote_addr=("10.255.255.255", helper_control_recv_port))
                elif msg_type == network.message.TYPE_ROUTE and self_ip != addr[0]:
                    helpee_id, route_bytes, seq_num = network.message.vehicle_parse_route_packet_data(data[-msg_size:])
                    if helpee_id != vehicle_id and (helpee_id not in vehicle_seq_dict.keys() or \
                            seq_num > vehicle_seq_dict[helpee_id]):
                        # helpee only rebroadcast route not equal to themselves and with larger seq
                        vehicle_seq_dict[helpee_id] = seq_num
                        network.message.send_msg(v2v_control_socket, data[:network.message.CONTROL_MSG_HEADER_LEN],\
                                        data[network.message.CONTROL_MSG_HEADER_LEN:], is_udp=True, \
                                        remote_addr=("10.255.255.255", helper_control_recv_port))
                elif msg_type == network.message.TYPE_SOS_REPLY and self_ip != addr[0]:
                    # get a helper reply that I can help you
                    helper_id = network.message.vehicle_parse_sos_packet_data(data[-msg_size:])
                    if current_helper_id == 65535:
                        # dont have a helper yet, take this one
                        helper_ip = "10.0.0." + str(helper_id+2)  
                        current_helper_id = helper_id 
                        new_send_thread = VehicleDataSendThread(helper_ip, helper_data_recv_port, helper_id)
                        new_send_thread.daemon = True
                        new_send_thread.start()
                        print("[Helpee decide to use helper assignment] helper_id: "\
                            + str(helper_id) + ' ' + str(time.time()), flush=True)
            else:
                print("Exception: no such connection state")


class VehicleDataRecvThread(threading.Thread):
    """Thread that handle data receiving between nodes (vehicles)
        Recv using tcp socket from helpee nodes
    """

    def __init__(self, helpee_socket, helpee_addr):
        threading.Thread.__init__(self)
        self.client_socket = helpee_socket
        self.client_address = helpee_addr
        self._is_closed = False
        self.helper_relay_server_sock = wwan.setup_p2p_links(vehicle_id, config.server_ip, 
                                                            config.server_data_port)
    

    def relay_ack_thread(self):
        while not self._is_closed:
            # recv and relay ack from server
            try:
                ack = self.helper_relay_server_sock.recv(2)
                self.client_socket.send(ack)
                print("[helper relay server ack] frame %d" % int.from_bytes(ack[:], 'big'))
            except:
                print("[Helpee already closed] skip relaying acks")

    def run(self):
        global v2v_recved_bytes
        ack_relay_thread = threading.Thread(target=self.relay_ack_thread)
        ack_relay_thread.start()
        while not self._is_closed:
            print('relay conn state: ', connection_state)
            data = b''
            header_to_recv = network.message.DATA_MSG_HEADER_LEN
            try:
                while len(data) < network.message.DATA_MSG_HEADER_LEN:
                    data_recv = self.client_socket.recv(header_to_recv)
                    data += data_recv
                    if len(data_recv) <= 0:
                        print("[Helpee closed]")
                        self._is_closed = True
                        self.helper_relay_server_sock.close()
                        return
                    header_to_recv -= len(data_recv)
                    v2v_recved_bytes += len(data_recv)
                msg_len, frame_id, v_id, type, _, _, _ = network.message.parse_data_msg_header(data)
                to_send = network.message.DATA_MSG_HEADER_LEN
                sent = 0
                while sent < to_send:
                    sent_len = self.helper_relay_server_sock.send(data[sent:])
                    sent += sent_len
                to_recv = msg_len
                curr_recv_bytes = 0
                t_start = time.time()
                while curr_recv_bytes < msg_len:
                    data = self.client_socket.recv(65536 if to_recv > 65536 else to_recv)
                    if len(data) <= 0:
                        print("[Helpee closed]")
                        self._is_closed = True
                        self.helper_relay_server_sock.close()
                        break
                    curr_recv_bytes += len(data)
                    to_recv -= len(data)
                    v2v_recved_bytes += len(data)
                    self.helper_relay_server_sock.send(data)
                t_elasped = time.time() - t_start
                est_v2v_thrpt = msg_len/t_elasped/1000000.0
                if self._is_closed:
                    return
                if type == TYPE_PCD:
                    print("[Received a full frame/oxts] %f frame: %d vehicle: %d %f" 
                            % (est_v2v_thrpt, frame_id, v_id, time.time()), flush=True)
            except:
                print("[Helpee closed the connection]")
                self._is_closed = True
            
        # self.client_socket.close()

    def stop():
        pass


class VehicleDataSendThread(threading.Thread):
    """ Thread that handle data sending between nodes (vehicles)
        Used when helpee get assignment and connect to the helper node
    """

    def __init__(self, helper_ip, helper_port, helper_id):
        global current_helper_id
        threading.Thread.__init__(self)
        self.v2v_data_send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.v2v_data_send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        try:
            self.v2v_data_send_sock.connect((helper_ip, helper_port))
            self.is_helper_alive = True
            self.ack_recv_thread = threading.Thread(target=self.ack_recv_thread, args=())
            current_helper_id = helper_id
        except:
            print('[Connection to helper failed]')

    
    def run(self):
        global curr_frame_id, last_frame_sent_ts, curr_frame_rate
        self.ack_recv_thread.start()
        while (self.is_helper_alive) and (connection_state == "Disconnected"):
            t_start = time.time()
            frame_lock.acquire()
            curr_f_id = curr_frame_id if not is_adaptive_frame_skipped else get_curr_tranmist_frame_id()
            # pcd = pcd_data_buffer[curr_frame_id % config.MAX_FRAMES]
            # pcd = pcd_data_buffer[curr_frame_id % config.MAX_FRAMES][0]
            pcd, num_chunks = get_encoded_frame(curr_frame_id, get_latency(e2e_frame_latency))
            # if pcd_data_type == "GTA":
            oxts = oxts_data_buffer[curr_frame_id % config.MAX_FRAMES]
            curr_frame_id += 1
            frame_lock.release()
            print("[V2V send pcd frame] Start sending frame " + str(curr_f_id) + " to helper " \
                    + str(current_helper_id) + ' ' + str(time.time()), flush=True)
            frame_sent_time[curr_f_id] = time.time()
            if_send_success = send(self.v2v_data_send_sock, pcd, curr_f_id, TYPE_PCD, num_chunks,\
                pcd_data_buffer[curr_f_id%config.MAX_FRAMES][:num_chunks])
            if not if_send_success:
                print("send not in time!!")
                self.stop()
            if_send_success = send(self.v2v_data_send_sock, oxts, curr_f_id, TYPE_OXTS)
            t_elapsed = time.time() - t_start
            if capture_finished and is_adaptive_frame_skipped:
                curr_frame_rate = get_updated_fps(get_latency(e2e_frame_latency))
                print("Update framerate: ", curr_frame_rate)
            if capture_finished and (1/curr_frame_rate - t_elapsed) > 0:
                time.sleep(1/curr_frame_rate - t_elapsed)

        print("[Change helper/reconnect to server] close the prev conn thread " + str(time.time()))
        self.v2v_data_send_sock.close()


    def ack_recv_thread(self):
        while self.is_helper_alive:
            try:
                # recv acknowledgement from server
                ack = self.v2v_data_send_sock.recv(2)
                frame_id = int.from_bytes(ack, 'big')
                recv_time = time.time()
                frame_latency = recv_time - frame_sent_time[frame_id]
                print("[Recv ack from helper relay] frame %d latency %f"%(frame_id,frame_latency))
                e2e_frame_latency_lock.acquire()
                e2e_frame_latency[frame_id] = frame_latency
                e2e_frame_latency_lock.release()
            except Exception as e:
                # socket might be closed since helper change
                print(e)
                print('[Helper changed] not recv ACK from prev helper anymore')
                return 


    def stop(self):
        self.is_helper_alive = False


def check_if_disconnected(disconnect_timestamps):
    """ Check whether the vehicle is disconnected to the server

    Args:
        disconnect_timestamps (dictionary): the disconnected timestamp dictionary read from
        LTE trace e.g. dict = {1: [2.1, 10.0]} means node 1 disconnects at 2.1 sec and reconnect at
        10.0 sec

    Returns:
        Bool: True for disconnected, False else
    """
    # print("check if V2I conn is disconnected")
    if vehicle_id in disconnect_timestamps.keys():
        elapsed_time = time.time() - start_timestamp
        index = bisect.bisect(disconnect_timestamps[vehicle_id], elapsed_time)
        if index % 2 == 1:
            # if index is an odd number, it falls in to the disconnected region 
            return True
        else:
            return False
    else:
        return False


def send_control_msgs(node_type):
    global control_seq_num
    # if node_type == HELPER:
    v2i_control_socket_lock.acquire()
    if scheduler_mode == 'distributed':
        if node_type == HELPEE:
            wlan.broadcast_sos_msg(v2v_control_socket, vehicle_id)
    if scheduler_mode == 'minDist' or scheduler_mode == 'combined' or scheduler_mode == 'bwAware' \
        or scheduler_mode == 'random':
        # send helpee/helper info/loc 
        if node_type == HELPER:
            wwan.send_location(HELPER, vehicle_id, self_loc, v2i_control_socket, control_seq_num)
        else:
            mobility.broadcast_location(vehicle_id, self_loc, v2v_control_socket, control_seq_num)
        control_seq_num += 1
    if scheduler_mode == 'combined' or scheduler_mode == 'routeAware':
        # send routing info
        if node_type == HELPER:
            wwan.send_route(HELPER, vehicle_id, route.table_to_bytes(route.get_routes(vehicle_id)), 
                            v2i_control_socket, control_seq_num)
        else:
            route.broadcast_route(vehicle_id, route.get_routes(vehicle_id), v2v_control_socket,\
                                control_seq_num)
        control_seq_num += 1
    # if node_type == HELPER:
    v2i_control_socket_lock.release()   
    
    
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
        if connection_state == "Connected" and not control_msg_disabled:
            # print("Connected to server...")
            # send out the corresponding control messages
            send_control_msgs(HELPER)
        elif connection_state == "Disconnected" and not control_msg_disabled:
            # print("Disconnected to server... broadcast " + str(time.time()))
            send_control_msgs(HELPEE)
        if check_if_disconnected(disconnect_timestamps):
            connection_state = "Disconnected"
        else:
            connection_state = "Connected"
        t_elasped = time.time() - t_start
        if 0.2-t_elasped > 0:
            time.sleep(0.2-t_elasped)



def main():
    global start_timestamp
    # senser_data_capture_thread = threading.Thread(target=sensor_data_capture, \
    #          args=(PCD_DATA_PATH, OXTS_DATA_PATH, FRAMERATE))
    # senser_data_capture_thread.start()
    trace_files = 'trace.txt'
    lte_traces = utils.read_traces(trace_files)
    disconnect_timestamps = utils.process_traces(lte_traces, HELPEE_CONF)
    print("disconnect timestamp")
    print(disconnect_timestamps)
    t_start = time.time()
    sensor_data_capture(PCD_DATA_PATH, OXTS_DATA_PATH, FRAMERATE)
    t_elapsed = time.time() - t_start
    print("read and encode takes %f" % t_elapsed)
    # explicitly sync on encoding
    if t_elapsed < 10:
        time.sleep(10-t_elapsed)
    start_timestamp = time.time()
    print("[start timestamp] ", start_timestamp)
    loction_update_thread = threading.Thread(target=self_loc_update_thread, args=())
    loction_update_thread.daemon = True
    loction_update_thread.start()
    v2i_control_thread = ServerControlThread()
    v2i_control_thread.start()
    v2v_control_thread = VehicleControlThread()
    v2v_control_thread.start()
    v2v_data_thread = threading.Thread(target=v2v_data_recv_thread, args=())
    v2v_data_thread.start()
    v2i_data_thread = threading.Thread(target=v2i_data_send_thread, args=())
    v2i_data_thread.start()

    throughput_thread = threading.Thread(target=throughput_calc_thread, args=())
    throughput_thread.daemon = True
    throughput_thread.start()

    check_connection_state(disconnect_timestamps)


if __name__ == '__main__':
    main()