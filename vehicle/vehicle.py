# this file handles the main process run by each vehicle node
# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stderr = sys.stdout  # redirect stderr to stdout for debugging
import argparse
import bisect
import socket
import threading
import time
import pickle
import config
import network.message
import ptcl.partition
import ptcl.pointcloud
import utils
import mobility
import route
import wlan
import wwan
import collections
import statistics as stats

from threading import Lock

s_print_lock = Lock()

def s_print(*a, **b):
    """Thread safe print function"""
    with s_print_lock:
        print(*a, **b)

# Define some constants
HELPEE = 0
HELPER = 1
TYPE_PCD = 0
TYPE_OXTS = 1
FRAMERATE = 5
PCD_ENCODE_LEVEL = 10  # point cloud encode level
PCD_QB = 11  # point cloud quantization bits
NO_ADAPTIVE_ENCODE = 0
ADAPTIVE_ENCODE = 1
ADAPTIVE_ENCODE_FULL_CHUNK = 2


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', default=0, type=int, help='vehicle id')
parser.add_argument('-d', '--data_path', default='~/DeepGTAV-data/object-0227-1/', \
                    type=str, help='point cloud and oxts data path')
parser.add_argument('--data_type', default="Carla", choices=["GTA", "Carla"])
parser.add_argument('-l', '--location_file',
                    default=os.path.dirname(os.path.abspath(__file__)) + "/input/object-0227-loc.txt",
                    type=str, help='location file name')
parser.add_argument('-c', '--helpee_conf',
                    default=os.path.dirname(os.path.abspath(__file__)) + "/input/helpee_conf/helpee-nodes.txt",
                    type=str, help='helpee nodes configuration file')
parser.add_argument('-f', '--fps', default=10, type=int, help='FPS of pcd data')
parser.add_argument('-n', '--disable_control', default=0, type=int, help='disable control msgs')
parser.add_argument('--adaptive', default=0, type=int,
                    help="adaptive encoding type (0 for no adaptive encoding, 1 for adaptive, 2 for adaptive but always use full 4 chunks")
parser.add_argument('--adapt_skip_frames', default=False, action="store_true",
                    help="enable adaptive frame skipping when sending takes too long")
parser.add_argument('--add_loc_noise', default=False, action='store_true',
                    help="enable noise to location")
parser.add_argument('--v2v_mode', default=0, type=int, choices=[0, 1])
parser.add_argument('-t', '--start_timestamp', default=time.time(), type=float)
args = parser.parse_args()

control_msg_disabled = True if args.disable_control == 1 else False
vehicle_id = args.id
is_adaptive_frame_skipped = args.adapt_skip_frames
add_noise_to_loc = args.add_loc_noise
s_print("Noise enabled?", add_noise_to_loc)

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
SERVER_IP = config.server_ip
if args.v2v_mode == 1:
    SERVER_IP = "10.0.0.2"
s_print('fps ' + str(FRAMERATE))
curr_frame_rate = FRAMERATE
connection_state = "Connected"
current_helpee_id = 65535
current_helper_id = 65535
start_timestamp = 0.0
last_frame_sent_ts = 0.0
vehicle_locs = mobility.read_locations(LOCATION_FILE)
self_loc_trace = vehicle_locs[vehicle_id]
if len(self_loc_trace) > 5:
    self_loc_trace = self_loc_trace[5:]  # manually sync location update with vechiular_perception.py
self_loc = self_loc_trace[0]
self_group = (0, 0)
pcd_data_buffer = []
oxts_data_buffer = []
pcd_data_buffer_adaptive = collections.defaultdict(list)
encoding_sizes = {}
e2e_frame_latency = {}
e2e_frame_latency_lock = threading.Lock()
frame_sent_time = {}


def self_loc_update_thread():
    """Thread to update self location by GPS
    """
    global self_loc
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', int(20175)))
    while True:
        data = s.recv(4096)
        if not data:
            break
        line = data.decode('utf-8')
        # print(line)
        split = line.split(',')
        if line.startswith('$GPGGA'):
            longitude, latitude = float(split[4])/100, float(split[2])/100
            # print(longitude, latitude, time.time())
            self_loc = (latitude, longitude)
        elif line.startswith('$GPRMC') and split[2] == 'A':
            longitude, latitude = float(split[5])/100, float(split[3])/100
            self_loc = (latitude, longitude)
            # print(longitude, latitude, time.time())


vehicle_seq_dict = {}  # store other vehicle's highest seq
control_seq_num = 0  # self seq number

frame_lock = threading.Lock()
v2i_control_socket_lock = threading.Lock()
v2v_no_route = False
try:
    v2i_control_socket, scheduler_mode = wwan.setup_p2p_links(vehicle_id, SERVER_IP,
                                                              config.server_ctrl_port, recv_sched_scheme=True)
    v2i_data_socket = wwan.setup_p2p_links(vehicle_id, SERVER_IP, config.server_data_port)
except OSError as e:
    s_print('no route to host in V2V mode')
    v2i_control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    v2v_no_route = True
    scheduler_mode = 'v2v'
s_print("server sched mode", scheduler_mode)
v2v_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
v2v_data_control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
curr_frame_id = 0
v2v_recved_bytes = 0

helper_data_send_thread = []
helper_data_recv_port = 8080
helper_control_recv_port = 8888
helpee_rst_recv_port = 8000  # this is the port that traffic should be prioritized for
self_ip = "10.0.0." + str(vehicle_id + 2)
capture_finished = False
fallback_socket_pool = {}
current_mode = scheduler_mode


def throughput_calc_thread():
    """ Thread to calculate V2V thrpt
    """
    global v2v_recved_bytes
    while True:
        s_print("[relay throughput] %f Mbps %f" % (v2v_recved_bytes * 8.0 / 1000000, time.time()))
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
            pcd_f_name = pcd_data_path + "%06d.bin" % i
            oxts_f_name = oxts_data_path + "%06d.txt" % i
        elif pcd_data_type == "Carla":
            pcd_f_name = pcd_data_path + str(1000 + i) + ".npy"
            oxts_f_name = oxts_data_path + str(1000 + i) + ".trans.npy"
        oxts_data_buffer.append(ptcl.pointcloud.read_oxts(oxts_f_name, pcd_data_type))
        pcd_np = ptcl.pointcloud.read_pointcloud(pcd_f_name, pcd_data_type)

        if ADAPTIVE_ENCODE_TYPE == NO_ADAPTIVE_ENCODE:
            partitioned = ptcl.partition.simple_partition(pcd_np, 50)
            # partitioned = pcd_np
            pcd, ratio = ptcl.pointcloud.dracoEncode(partitioned, PCD_ENCODE_LEVEL, PCD_QB)
            pcd_data_buffer.append(pcd)
        else:
            partitioned = ptcl.partition.simple_partition(pcd_np, 50)
            for qb in range(7, PCD_QB + 1):
                encoded, ratio = ptcl.pointcloud.dracoEncode(partitioned, PCD_ENCODE_LEVEL, qb)
                pcd_data_buffer_adaptive[qb].append(encoded)
            # pcd_data_buffer.append(pcd_np)
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
        s_print("frame id: " + str(frame_id) + " qb: " + str(PCD_QB))
        s_print("frame id: " + str(frame_id) + " latency: " + str(metric) + " number of chunks: " + str(1))
    elif ADAPTIVE_ENCODE_TYPE == ADAPTIVE_ENCODE:
        # TODO: change different # of chunks to different encoding levels
        # frame = pcd_data_buffer[frame_id % config.MAX_FRAMES]
        cnt = 1
        qb = get_encoding_qb(e2e_frame_latency)
        encoded_frame = pcd_data_buffer_adaptive[qb][frame_id % config.MAX_FRAMES]
        encoding_sizes[frame_id] = len(encoded_frame)
        s_print("frame id: " + str(frame_id) + " qb: " + str(qb))
        s_print("frame id: " + str(frame_id) + " latency: " + str(metric) + " number of chunks: " + str(cnt))
    elif ADAPTIVE_ENCODE_TYPE == ADAPTIVE_ENCODE_FULL_CHUNK:
        cnt = 4
        encoded_frame = pcd_data_buffer[frame_id % config.MAX_FRAMES][0] + \
                        pcd_data_buffer[frame_id % config.MAX_FRAMES][1] + \
                        pcd_data_buffer[frame_id % config.MAX_FRAMES][2] + \
                        pcd_data_buffer[frame_id % config.MAX_FRAMES][3]
        s_print("frame id: " + str(frame_id) + " latency: " + str(metric) + " number of chunks: " + str(4))
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


def get_encoding_qb(e2e_frame_latency):
    if len(encoding_sizes) == 0 or len(e2e_frame_latency) == 0:
        return PCD_QB
    e2e_frame_latency_lock.acquire()
    recent_latencies = sorted(e2e_frame_latency.items(), key=lambda item: -item[0])
    e2e_frame_latency_lock.release()
    cnt = 0
    thrpts = []
    for id, latency in recent_latencies:
        if id in encoding_sizes:
            if latency < 0:
                s_print('err!!! latency < 0')
                latency = 0.4
            thrpt = encoding_sizes[id] / latency
            # print('thrpt:', thrpt)
            thrpts.append(thrpt)
            cnt += 1
            if cnt == 5:
                break
        # if cnt <= len(encoding_sizes):
        #     thrpt = encoding_sizes[-cnt] / latency
        #     thrpts.append(thrpt)
    avg_thrpt = stats.harmonic_mean(thrpts) * 8 / 1000000.
    s_print('Avg thrpt for past 5 chunks:', avg_thrpt)
    return map_thrpt_to_encoding_level(avg_thrpt)


def map_thrpt_to_encoding_level(thrpt):
    # if thrpt >= 9.6:
    #     return 12
    # el
    if thrpt >= 6.24:
        return 11
    elif thrpt >= 3.2:
        return 10
    elif thrpt >= 1.2:
        return 9
    elif thrpt >= 0.64:
        return 8
    else:
        return 7


def send(socket, data, id, type, num_chunks=1, chunks=None):
    msg_len = len(data)
    frame_ready_timestamp = start_timestamp + id * 1.0 / FRAMERATE
    header = network.message.construct_data_msg_header(data, type, id, vehicle_id, frame_ready_timestamp, num_chunks,
                                                       chunks)
    s_print("[send header] vehicle %d, frame %d, data len: %d" % (vehicle_id, id, msg_len))
    hender_sent = 0
    while hender_sent < len(header):
        try:
            bytes_sent = socket.send(header[hender_sent:])
            hender_sent += bytes_sent
        except:
            s_print('[Send error]')
            return False
    total_sent = 0
    while total_sent < msg_len:
        try:
            bytes_sent = socket.send(data[total_sent:])
            s_print("[Sedning Data] Sent %d bytes" % bytes_sent)
            total_sent += bytes_sent
        except:
            s_print('[Send error]')
            return False
    return True


def get_frame_ready_timestamp(frame_id, fps):
    return start_timestamp + frame_id * (1 / fps)


def get_curr_tranmist_frame_id():
    return int((time.time() - start_timestamp) * FRAMERATE)


def get_time_to_next_ready_frame():
    offset = time.time() - start_timestamp


def v2i_data_send_thread():
    """Thread to handle V2I data sending
    """
    global curr_frame_id, last_frame_sent_ts, curr_frame_rate
    ack_recv_thread = threading.Thread(target=v2i_ack_recv_thread, args=(v2i_data_socket,))
    ack_recv_thread.start()
    while True:
        if connection_state == "Connected":
            t_start = time.time()
            frame_lock.acquire()
            curr_f_id = curr_frame_id if not is_adaptive_frame_skipped else get_curr_tranmist_frame_id()
            expected_trasmit_frame = get_curr_tranmist_frame_id()
            s_print("Current sending frame %d, latest ready frame %d" % (curr_f_id, expected_trasmit_frame))
            if curr_f_id > expected_trasmit_frame:
                frame_lock.release()
                next_ready_frame_time = (time.time() - start_timestamp) % (1.0 / curr_frame_rate)
                time.sleep(1 / curr_frame_rate - next_ready_frame_time)
                continue
            data_f_id = curr_f_id % config.MAX_FRAMES
            pcd, num_chunks = get_encoded_frame(curr_frame_id, get_latency(e2e_frame_latency))
            curr_frame_id += 1
            frame_lock.release()
            last_frame_sent_ts = time.time()
            # last_frame_sent_ts should be the timestamp for finishing the sensor data capture
            last_frame_sent_ts = get_frame_ready_timestamp(curr_f_id, FRAMERATE)
            s_print("[V2I send pcd frame] " + str(curr_f_id) + ' ' + str(last_frame_sent_ts), flush=True)
            frame_sent_time[curr_f_id] = time.time()
            # check if fallbacked
            if current_mode != 'pure-V2V':
                send_soc = v2i_data_socket
            else:
                send_soc = fallback_socket_pool['server-data']
            send(send_soc, pcd, curr_f_id, TYPE_PCD, num_chunks, pcd)
            oxts = oxts_data_buffer[data_f_id]
            send(send_soc, oxts, curr_f_id, TYPE_OXTS)
            t_elapsed = time.time() - t_start
            if capture_finished and is_adaptive_frame_skipped:
                curr_frame_rate = get_updated_fps(get_latency(e2e_frame_latency))
                s_print("Update framerate: ", curr_frame_rate, time.time())
            if capture_finished and (1.0 / curr_frame_rate - t_elapsed) > 0 \
                    and expected_trasmit_frame == curr_f_id:
                # only sleep and wait if current sending frame is matched to the newest ready frame
                s_print("capture finished, sleep %f" % (1.0 / curr_frame_rate - t_elapsed))
                time_to_next_ready_frame = (time.time() - start_timestamp) % (1.0 / curr_frame_rate)
                time.sleep(1.0 / curr_frame_rate - time_to_next_ready_frame)
            elif (1.0 / curr_frame_rate - t_elapsed) < 0:
                s_print('sending is taking too long!')

            # if curr_frame_rate != 10:
            #     frame_lock.acquire()
            #     curr_frame_id += int(10/curr_frame_rate)
            #     frame_lock.release()
        else:
            time.sleep(0.1)


def v2i_ack_recv_thread(soc):
    while True:
        header, payload = network.message.recv_msg(soc, network.message.TYPE_SERVER_REPLY_MSG)
        payload_size, frame_id, msg_type, ts = network.message.parse_server_reply_msg_header(header)
        downlink_latency = time.time() - ts
        if msg_type == network.message.TYPE_SEVER_ACK_MSG:
            frame_latency = time.time() - frame_sent_time[frame_id]
            s_print("[Recv ack from server] frame %d, latency %f, dl latency %f" %
                  (frame_id, frame_latency, downlink_latency), time.time())
            e2e_frame_latency_lock.acquire()
            if frame_latency - downlink_latency < 0
                e2e_frame_latency[frame_id] = frame_latency
            else:
                e2e_frame_latency[frame_id] = frame_latency - downlink_latency
            e2e_frame_latency_lock.release()
        elif msg_type == network.message.TYPE_SERVER_REPLY_MSG:
            s_print("[Recv rst from server] frame %d, DL latency %f" % (frame_id, downlink_latency),
                  time.time())


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


def v2v_data_recv_thread(port=helper_data_recv_port):
    """ a seperate thread to recv point cloud from helpee node and forward it to the server
    """
    host_ip = ''
    host_port = port
    v2v_data_recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    v2v_data_recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    v2v_data_recv_sock.bind((host_ip, host_port))
    while True:
        v2v_data_recv_sock.listen()
        client_socket, client_address = v2v_data_recv_sock.accept()
        s_print("[get helpee connection] " + str(time.time()))
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
    helpee_addr = "10.0.0." + str(helpee_id + 2)
    # helpee_addr = "10.42.0.163"
    network.message.send_msg(send_note_sock, header, msg, is_udp=True,
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
            global current_helpee_id, self_group, current_mode
            data, msg_type = wwan.recv_control_msg(v2i_control_socket)

            if is_helper_recv() and msg_type == network.message.TYPE_ASSIGNMENT:
                if if_recv_nothing_from_server(data):
                    s_print("Recv nothing from server " + str(time.time()))
                elif if_not_assigned_as_helper(data):
                    s_print("Not assigned as helper.. " + str(time.time()))
                else:
                    s_print("[Helper get assignment from server] helpee_id: " + \
                          str(data) + ' ' + str(time.time()))
                    # print(self_loc)
                    current_helpee_id = data
                    notify_helpee_node(current_helpee_id)
            elif msg_type == network.message.TYPE_GROUP:
                self_group = data
                if is_helper_recv() and current_helpee_id != 65535:
                    s_print("notify helpee node about group change", data)
                    send_note_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                    payload = pickle.dumps(data)
                    header = network.message.construct_control_msg_header(payload, network.message.TYPE_GROUP)
                    helpee_addr = "10.0.0." + str(current_helpee_id + 2)
                    # helpee_addr = "10.42.0.163"
                    network.message.send_msg(send_note_sock, header, payload, is_udp=True, \
                                             remote_addr=(helpee_addr, helper_control_recv_port))
            elif msg_type == network.message.TYPE_FALLBACK:
                # fallback to V2V
                server_ip = config.v2v_server_ip
                if current_mode != "pure-V2V":
                    # fallback_socket_pool['server-ctrl'] = wwan.setup_p2p_links(vehicle_id, server_ip, \
                    #                 config.server_ctrl_port, recv_sched_scheme=True)[0]
                    fallback_socket_pool['server-data'] = wwan.setup_p2p_links(vehicle_id, server_ip, \
                                                                               config.server_data_port)
                    current_mode = "pure-V2V"
                    ack_recv_thread = threading.Thread(target=v2i_ack_recv_thread,
                                                       args=(fallback_socket_pool['server-data'],))
                    ack_recv_thread.start()
            elif msg_type == network.message.TYPE_RECONNECT:
                if current_mode == "pure-V2V":
                    current_mode = scheduler_mode


class VehicleControlThread(threading.Thread):
    """Thread that handle control messages between nodes (vehicles)
    """

    def __init__(self):
        threading.Thread.__init__(self)
        s_print("Setup V2V socket to recv broadcast message")
        global v2v_control_socket
        host_ip = ''
        host_port = helper_control_recv_port
        # Use UDP socket for broadcasting
        v2v_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        v2v_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        v2v_control_socket.bind((host_ip, host_port))

    def run(self):
        global current_helper_id, control_seq_num, self_group
        while True:
            data, addr = network.message.recv_msg(v2v_control_socket, network.message.TYPE_CONTROL_MSG, \
                                                  is_udp=True)
            msg_size, msg_type = network.message.parse_control_msg_header(data)
            if connection_state == "Connected":
                # This vehicle is a helper now
                if msg_type == network.message.TYPE_LOCATION:
                    s_print("[helper recv location broadcast] " + str(time.time()))
                    helpee_id, helpee_loc, seq_num, group_id = network.message.vehicle_parse_location_packet_data(
                        data[-msg_size:])
                    if group_id != self_group:  # skip if not in the same group
                        continue
                    # send helpee location
                    s_print((helpee_id, helpee_loc))
                    if helpee_id not in vehicle_seq_dict.keys() \
                            or seq_num > vehicle_seq_dict[helpee_id]:
                        # only send location if received seq num is larger
                        vehicle_seq_dict[helpee_id] = seq_num
                        v2i_control_socket_lock.acquire()
                        wwan.send_location(HELPEE, helpee_id, helpee_loc, v2i_control_socket, seq_num, add_noise=False)
                        control_seq_num += 1
                        v2i_control_socket_lock.release()
                elif msg_type == network.message.TYPE_ROUTE:
                    s_print("[helper recv route broadcast] " + str(time.time()))
                    helpee_id, route_bytes, seq_num, group_id = network.message.vehicle_parse_route_packet_data(
                        data[-msg_size:])
                    if group_id != self_group:  # skip if not in the same group
                        continue
                    # forward helpee route
                    if helpee_id not in vehicle_seq_dict.keys() \
                            or seq_num > vehicle_seq_dict[helpee_id]:
                        s_print("[helper forwarding route broadcast] " + str(time.time()))
                        # only send location if received seq num is larger
                        vehicle_seq_dict[helpee_id] = seq_num
                        v2i_control_socket_lock.acquire()
                        wwan.send_route(HELPEE, helpee_id, route_bytes, v2i_control_socket, seq_num)
                        control_seq_num += 1
                        v2i_control_socket_lock.release()
                elif msg_type == network.message.TYPE_SOS and self_ip != addr[0]:
                    # helpee run in distributed way, send back help msg
                    s_print("[helper recv sos broadcast] " + str(addr) + str(time.time()))
                    wlan.echo_sos_msg(v2v_control_socket, vehicle_id, addr)
                elif msg_type == network.message.TYPE_GROUP:
                    pass
            elif connection_state == "Disconnected":
                # This vehicle is a helpee now
                if msg_type == network.message.TYPE_ASSIGNMENT:
                    helper_id = int.from_bytes(data[-msg_size:], 'big')
                    s_print("[Helpee get helper assignment] helper_id: " \
                          + str(helper_id) + ' ' + str(time.time()), flush=True)
                    if helper_id != current_helper_id:
                        helper_ip = "10.0.0." + str(helper_id + 2)
                        new_send_thread = VehicleDataSendThread(helper_ip, helper_data_recv_port, helper_id)
                        new_send_thread.daemon = True
                        new_send_thread.start()
                        if len(helper_data_send_thread) != 0:
                            helper_data_send_thread[-1].stop()
                        helper_data_send_thread.append(new_send_thread)
                elif msg_type == network.message.TYPE_LOCATION and self_ip != addr[0]:
                    helpee_id, helpee_loc, seq_num, group_id = network.message.vehicle_parse_location_packet_data(
                        data[-msg_size:])
                    if group_id != self_group:  # skip if not in the same group
                        continue
                    if helpee_id != vehicle_id and (helpee_id not in vehicle_seq_dict.keys() or \
                                                    seq_num > vehicle_seq_dict[helpee_id]):
                        # helpee only rebroadcast loc not equal to themselves and with larger seq
                        vehicle_seq_dict[helpee_id] = seq_num
                        network.message.send_msg(v2v_control_socket, data[:network.message.CONTROL_MSG_HEADER_LEN], \
                                                 data[network.message.CONTROL_MSG_HEADER_LEN:], is_udp=True, \
                                                 remote_addr=("10.255.255.255", helper_control_recv_port))
                elif msg_type == network.message.TYPE_ROUTE and self_ip != addr[0]:
                    helpee_id, route_bytes, seq_num, group_id = network.message.vehicle_parse_route_packet_data(
                        data[-msg_size:])
                    if group_id != self_group:  # skip if not in the same group
                        continue
                    if helpee_id != vehicle_id and (helpee_id not in vehicle_seq_dict.keys() or \
                                                    seq_num > vehicle_seq_dict[helpee_id]):
                        # helpee only rebroadcast route not equal to themselves and with larger seq
                        vehicle_seq_dict[helpee_id] = seq_num
                        network.message.send_msg(v2v_control_socket, data[:network.message.CONTROL_MSG_HEADER_LEN], \
                                                 data[network.message.CONTROL_MSG_HEADER_LEN:], is_udp=True, \
                                                 remote_addr=("10.255.255.255", helper_control_recv_port))
                elif msg_type == network.message.TYPE_SOS_REPLY and self_ip != addr[0]:
                    # get a helper reply that I can help you
                    helper_id = network.message.vehicle_parse_sos_packet_data(data[-msg_size:])
                    if current_helper_id == 65535:
                        # dont have a helper yet, take this one
                        helper_ip = "10.0.0." + str(helper_id + 2)
                        current_helper_id = helper_id
                        new_send_thread = VehicleDataSendThread(helper_ip, helper_data_recv_port, helper_id)
                        new_send_thread.daemon = True
                        new_send_thread.start()
                        s_print("[Helpee decide to use helper assignment] helper_id: " \
                              + str(helper_id) + ' ' + str(time.time()), flush=True)
                elif msg_type == network.message.TYPE_GROUP:
                    group_id = pickle.loads(data)
                    self_group = group_id
                    s_print('[helpee update group id]', group_id)

            else:
                s_print("Exception: no such connection state")


class VehicleDataControlRecvThread(threading.Thread):
    """Thread that handle control messages between nodes (vehicles)
    """

    def __init__(self):
        threading.Thread.__init__(self)
        s_print("Setup V2V socket to recv data control message")
        global v2v_data_control_socket
        host_ip = ''
        host_port = 8000
        v2v_data_control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # v2v_data_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        v2v_data_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        v2v_data_control_socket.bind((host_ip, host_port))

    def run(self):
        while True:
            # udp type
            # try:
            #     header, payload = network.message.recv_msg(v2v_data_control_socket, network.message.TYPE_SERVER_REPLY_MSG, is_udp=True)
            #     payload_size, frame_id, msg_type, ts = network.message.parse_server_reply_msg_header(header)
            #     downlink_latency = time.time() - ts
            #     if msg_type == network.message.TYPE_SEVER_ACK_MSG:
            #         frame_latency = time.time() - frame_sent_time[frame_id]
            #         print("[V2V Recv ack from helper] frame %d, latency %f, DL latency %f"
            #             %(frame_id, frame_latency, downlink_latency))
            #         e2e_frame_latency_lock.acquire()
            #         e2e_frame_latency[frame_id] = frame_latency
            #         e2e_frame_latency_lock.release()
            #     elif msg_type == network.message.TYPE_SERVER_REPLY_MSG:
            #         print("[V2V Recv rst from helper] frame %d, DL latency %f"%
            #         (frame_id, downlink_latency), time.time())            
            # except Exception as e:
            #     print("Exception:", e)
            #     print('[Helper changed] not recv ACK from prev helper anymore')
            #     return
            # tcp type
            v2v_data_control_socket.listen()
            helper_socket, helper_address = v2v_data_control_socket.accept()
            s_print("[Start recv V2V ACKs] Get connection from", helper_address)
            new_data_recv_thread = threading.Thread(target=self.ack_recv_thread, args=(helper_socket,))
            new_data_recv_thread.start()

    def ack_recv_thread(self, helper_socket):
        while True:
            try:
                # recv acknowledgement from server
                header, payload = network.message.recv_msg(helper_socket, network.message.TYPE_SERVER_REPLY_MSG)
                # s_print('[V2V Recv Data Control Msg]')
                payload_size, frame_id, msg_type, ts = network.message.parse_server_reply_msg_header(header)
                downlink_latency = time.time() - ts
                if msg_type == network.message.TYPE_SEVER_ACK_MSG:
                    frame_latency = time.time() - frame_sent_time[frame_id]
                    s_print("[V2V Recv ack from helper] frame %d, latency %f, DL latency %f"
                          % (frame_id, frame_latency, downlink_latency))
                    e2e_frame_latency_lock.acquire()
                    if frame_latency < downlink_latency:
                        s_print(frame_sent_time[frame_id], ts)
                        s_print("err in dl latency?")
                    e2e_frame_latency[frame_id] = frame_latency - downlink_latency
                    e2e_frame_latency_lock.release()
                elif msg_type == network.message.TYPE_SERVER_REPLY_MSG:
                    s_print("[V2V Recv rst from helper] frame %d, DL latency %f" %
                          (frame_id, downlink_latency), time.time())

            except Exception as e:
                # socket might be closed since helper change
                s_print("Exception:", e)
                s_print('[Helper changed] not recv ACK from prev helper anymore')
                return


class VehicleDataRecvThread(threading.Thread):
    """Thread that handle data receiving between nodes (vehicles)
        Recv using tcp socket from helpee nodes
    """

    def __init__(self, helpee_socket, helpee_addr):
        threading.Thread.__init__(self)
        self.client_socket = helpee_socket
        self.client_address = helpee_addr
        self._is_closed = False
        self.helper_relay_server_sock = wwan.setup_p2p_links(vehicle_id, SERVER_IP,
                                                             config.server_data_port)
        self.client_ack_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_ack_socket.connect((helpee_addr[0], 8000))

    def relay_ack_thread(self):
        while not self._is_closed:
            # recv and relay ack from server
            try:
                header, payload = network.message.recv_msg(self.helper_relay_server_sock,
                                                           network.message.TYPE_SERVER_REPLY_MSG)
                # tcp type relay
                network.message.send_msg(self.client_ack_socket, header, payload)
                # udp type reply
                # send_note_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                # network.message.send_msg(send_note_sock, header, payload, is_udp=True,\
                #                     remote_addr=(self.client_address[0], 8000))
                s_print("[helper relay server ack] frame %d, %f" % (int.from_bytes(header[4:6], 'big'),
                                                                  time.time()))
            except Exception as e:
                s_print("[Helpee already closed] skip relaying acks", e)

    def run(self):
        global v2v_recved_bytes
        ack_relay_thread = threading.Thread(target=self.relay_ack_thread)
        ack_relay_thread.start()
        while not self._is_closed:
            # s_print('relay conn state: ', connection_state)
            data = b''
            header_to_recv = network.message.DATA_MSG_HEADER_LEN
            try:
                while len(data) < network.message.DATA_MSG_HEADER_LEN:
                    data_recv = self.client_socket.recv(header_to_recv)
                    data += data_recv
                    if len(data_recv) <= 0:
                        s_print("[Helpee closed]")
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
                        s_print("[Helpee closed]")
                        self._is_closed = True
                        self.helper_relay_server_sock.close()
                        break
                    curr_recv_bytes += len(data)
                    to_recv -= len(data)
                    v2v_recved_bytes += len(data)
                    self.helper_relay_server_sock.send(data)
                t_elasped = time.time() - t_start
                est_v2v_thrpt = msg_len / t_elasped / 1000000.0
                if self._is_closed:
                    return
                if type == TYPE_PCD:
                    s_print("[Received a full frame/oxts] %f frame: %d vehicle: %d %f"
                          % (est_v2v_thrpt, frame_id, v_id, time.time()), flush=True)
            except Exception as e:
                s_print("[Helpee closed the connection]", e)
                self._is_closed = True

        # self.client_socket.close()

    def stop(self):
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
            self.ack_thread = threading.Thread(target=self.ack_recv_thread, args=())
            current_helper_id = helper_id
        except:
            s_print('[Connection to helper failed]')

    def run(self):
        global curr_frame_id, last_frame_sent_ts, curr_frame_rate
        while (self.is_helper_alive) and (connection_state == "Disconnected"):
            t_start = time.time()
            frame_lock.acquire()
            curr_f_id = curr_frame_id if not is_adaptive_frame_skipped else get_curr_tranmist_frame_id()
            expected_trasmit_frame = get_curr_tranmist_frame_id()
            s_print("Current sending frame %d, latest ready frame %d" % (curr_f_id, expected_trasmit_frame))
            if curr_f_id > expected_trasmit_frame:
                frame_lock.release()
                next_ready_frame_time = (time.time() - start_timestamp) % (1.0 / curr_frame_rate)
                time.sleep(1.0 / curr_frame_rate - next_ready_frame_time)
                # time.sleep(0.01)
                continue
            pcd, num_chunks = get_encoded_frame(curr_frame_id, get_latency(e2e_frame_latency))
            oxts = oxts_data_buffer[curr_frame_id % config.MAX_FRAMES]
            curr_frame_id += 1
            frame_lock.release()
            s_print("[V2V send pcd frame] Start sending frame " + str(curr_f_id) + " to helper " \
                  + str(current_helper_id) + ' ' + str(time.time()), flush=True)
            frame_sent_time[curr_f_id] = time.time()
            if_send_success = send(self.v2v_data_send_sock, pcd, curr_f_id, TYPE_PCD, num_chunks, \
                                   pcd)
            if not if_send_success:
                s_print("send not in time!!")
                self.stop()
            if_send_success = send(self.v2v_data_send_sock, oxts, curr_f_id, TYPE_OXTS)
            t_elapsed = time.time() - t_start
            if capture_finished and is_adaptive_frame_skipped:
                curr_frame_rate = get_updated_fps(get_latency(e2e_frame_latency))
                s_print("Update framerate: ", curr_frame_rate)
            if capture_finished and (1 / curr_frame_rate - t_elapsed) > 0 \
                    and expected_trasmit_frame == curr_f_id:
                # only sleep and wait if current sending frame is matched to the newest ready frame
                time_to_next_ready_frame = (time.time() - start_timestamp) % (1.0 / curr_frame_rate)
                time.sleep(1 / curr_frame_rate - time_to_next_ready_frame)

        s_print("[Change helper/reconnect to server] close the prev conn thread " + str(time.time()))
        self.v2v_data_send_sock.close()

    def ack_recv_thread(self):
        while self.is_helper_alive:
            try:
                # recv acknowledgement from server
                header, payload = network.message.recv_msg(self.v2v_data_send_sock,
                                                           network.message.TYPE_SERVER_REPLY_MSG)
                payload_size, frame_id, msg_type, ts = network.message.parse_server_reply_msg_header(header)
                downlink_latency = time.time() - ts
                if msg_type == network.message.TYPE_SEVER_ACK_MSG:
                    frame_latency = time.time() - frame_sent_time[frame_id]
                    s_print("[Recv ack from server] frame %d, latency %f" % (frame_id, frame_latency))
                    e2e_frame_latency_lock.acquire()
                    if frame_latency - downlink_latency < 0:
                        e2e_frame_latency[frame_id] = frame_latency
                    else:
                        e2e_frame_latency[frame_id] = frame_latency - downlink_latency
                    e2e_frame_latency_lock.release()
                elif msg_type == network.message.TYPE_SERVER_REPLY_MSG:
                    s_print("[Recv rst from server] frame %d, DL latency %f" % (frame_id, downlink_latency))

            except Exception as e:
                # socket might be closed since helper change
                s_print(e)
                s_print('[Helper changed] not recv ACK from prev helper anymore')
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
    # s_print("check if V2I conn is disconnected")
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
    if scheduler_mode == 'v2i' or scheduler_mode == 'v2v':
        return
    v2i_control_socket_lock.acquire()
    if scheduler_mode == 'distributed':
        if node_type == HELPEE:
            wlan.broadcast_sos_msg(v2v_control_socket, vehicle_id)
    if scheduler_mode == 'minDist' or scheduler_mode == 'combined' or scheduler_mode == 'bwAware' \
            or scheduler_mode == 'random':
        # send helpee/helper info/loc 
        if node_type == HELPER:
            wwan.send_location(HELPER, vehicle_id, self_loc, v2i_control_socket, control_seq_num,
                               add_noise=add_noise_to_loc)
        else:
            mobility.broadcast_location(vehicle_id, self_loc, v2v_control_socket, control_seq_num, self_group,
                                        add_noise=add_noise_to_loc)
        control_seq_num += 1
    if scheduler_mode == 'combined' or scheduler_mode == 'routeAware':
        # send routing info
        if node_type == HELPER:
            wwan.send_route(HELPER, vehicle_id, route.table_to_bytes(route.get_routes(vehicle_id)),
                            v2i_control_socket, control_seq_num)
        else:
            route.broadcast_route(vehicle_id, route.get_routes(vehicle_id), v2v_control_socket, control_seq_num,
                                  self_group)
        control_seq_num += 1
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
        if check_if_disconnected(disconnect_timestamps):
            if connection_state == "Connected":
                s_print("disconnect to server ", time.time())
            connection_state = "Disconnected"
        else:
            connection_state = "Connected"
        if connection_state == "Connected" and not control_msg_disabled:
            # s_print("Connected to server...")
            # send out the corresponding control messages
            send_control_msgs(HELPER)
        elif connection_state == "Disconnected" and not control_msg_disabled:
            # s_print("Disconnected to server... broadcast " + str(time.time()))
            send_control_msgs(HELPEE)
        t_elasped = time.time() - t_start
        if 0.2 - t_elasped > 0:
            time.sleep(0.2 - t_elasped)


def main():
    global start_timestamp
    while time.time() < args.start_timestamp:
        time.sleep(0.005)
    disconnect_timestamps = utils.process_traces(HELPEE_CONF)

    if args.v2v_mode == 1:
        s_print("V2V enabled")
        disconnect_timestamps = {}
    else:
        s_print("disconnect timestamp")
        s_print(disconnect_timestamps)
    t_start = time.time()
    sensor_data_capture(PCD_DATA_PATH, OXTS_DATA_PATH, FRAMERATE)
    t_elapsed = time.time() - t_start
    s_print("read and encode takes %f" % t_elapsed)
    # explicitly sync on encoding
    if t_elapsed < 30:
        time.sleep(30 - t_elapsed)
    start_timestamp = time.time()
    s_print("[start timestamp] ", start_timestamp)
    loction_update_thread = threading.Thread(target=self_loc_update_thread, args=())
    loction_update_thread.daemon = True
    loction_update_thread.start()
    if not v2v_no_route:
        v2i_control_thread = ServerControlThread()
        v2i_control_thread.daemon = True
        v2i_control_thread.start()
        v2i_data_thread = threading.Thread(target=v2i_data_send_thread, args=())
        v2i_data_thread.daemon = True
        v2i_data_thread.start()
    if args.v2v_mode == 0:
        v2v_control_thread = VehicleControlThread()
        v2v_control_thread.daemon = True
        v2v_control_thread.start()
        v2v_data_control_thread = VehicleDataControlRecvThread()
        v2v_data_control_thread.start()
        v2v_data_thread = threading.Thread(target=v2v_data_recv_thread, args=())
        v2v_data_thread.daemon = True
        v2v_data_thread.start()

    # throughput_thread = threading.Thread(target=throughput_calc_thread, args=())
    # throughput_thread.daemon = True
    # throughput_thread.start()

    check_connection_state(disconnect_timestamps)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        s_print("Keyboard interrupted.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
