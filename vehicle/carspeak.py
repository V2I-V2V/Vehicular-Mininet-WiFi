from multiprocessing import Lock
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stderr = sys.stdout # redirect stderr to stdout for debugging
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
import wlan
import wwan
import collections
import statistics as stats
from octree import OcTree

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', default=0, type=int, help='vehicle id')
parser.add_argument('-d', '--data_path', default='~/DeepGTAV-data/object-0227-1/',\
                    type=str, help='point cloud and oxts data path')
parser.add_argument('--data_type', default="GTA", choices=["GTA", "Carla"])
parser.add_argument('-l', '--location_file', default=os.path.dirname(os.path.abspath(__file__)) + "/input/object-0227-loc.txt", \
                    type=str, help='location file name')
parser.add_argument('-f', '--fps', default=10, type=int, help='FPS of pcd data')
parser.add_argument('--start_timestamp', default=time.time(), type=float)
args = parser.parse_args()

vehicle_id = args.id
pcd_data_type = args.data_type
NODE_LEFT_TIMEOUT = 0.5
if args.data_type == "GTA":
    PCD_DATA_PATH = args.data_path + '/velodyne_2/'
    OXTS_DATA_PATH = args.data_path + '/oxts/'
else:
    PCD_DATA_PATH = args.data_path
    OXTS_DATA_PATH = args.data_path
    
LOCATION_FILE = args.location_file
FRAMERATE = args.fps
start_timestamp = 0.0
vehicle_locs = mobility.read_locations(LOCATION_FILE)
self_loc_trace = vehicle_locs[vehicle_id%len(vehicle_locs)]
if len(self_loc_trace) > 5:
    self_loc_trace = self_loc_trace[5:] # manually sync location update with vechiular_perception.py
self_loc = self_loc_trace[0]
pcd_data_buffer = []
oxts_data_buffer = []
self_ip = "10.0.0." + str(vehicle_id+2)
curr_frame_id = 0
last_frame_sent_ts = 0.0
host_ip = ''
host_port = 8888
v2v_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
v2v_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 145537)  
v2v_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
v2v_socket.bind((host_ip, host_port))
current_connected_vehicles, fully_finished_frame = {}, {}
connected_vehicle_lock = threading.Lock()


def self_loc_update_thread():
    """Thread to update self location every 100ms
    """
    global self_loc
    print("[start loc update] at %f" % time.time())
    for loc in self_loc_trace:
        t_s = time.time()
        self_loc = loc
        t_passed = time.time() - t_s
        if t_passed < 0.1:
            time.sleep(0.1-t_passed)
            

def get_encoded_frame_packets(frame_id):
    # ptcl_data = pcd_data_buffer[frame_id]
    # packets = OcTree.loss_resillient_encode(ptcl_data)
    packets = pcd_data_buffer[frame_id]
    return packets


def broadcast_sensor_data():
    global curr_frame_id, last_frame_sent_ts
    data_recv_thread = threading.Thread(target=sensor_data_recv, args=(v2v_socket,))
    data_recv_thread.start()
    while True:
        t_start = time.time()
        curr_f_id = curr_frame_id
        expected_trasmit_frame = get_curr_tranmist_frame_id()
        print("Current sending frame %d, latest ready frame %d"%(curr_f_id, expected_trasmit_frame))
        if curr_f_id > expected_trasmit_frame:
            next_ready_frame_time = (time.time() - start_timestamp) % (1.0/FRAMERATE)
            time.sleep(1/FRAMERATE - next_ready_frame_time)
            continue
        # elif expected_trasmit_frame > curr_f_id:
        #     curr_f_id = expected_trasmit_frame
        #     curr_frame_id = expected_trasmit_frame
        # else:
        curr_frame_id += 1
            
        last_frame_sent_ts = get_frame_ready_timestamp(curr_f_id, FRAMERATE)
        print("[V2I send pcd frame] " + str(curr_f_id) + ' ' + str(last_frame_sent_ts) + ' ' + str(time.time()), flush=True)
        data_f_id = curr_f_id % config.MAX_FRAMES
        packets_to_send = get_encoded_frame_packets(data_f_id)
        
        broadcast_packets(packets_to_send, v2v_socket, curr_f_id)
               
        t_elapsed = time.time() - t_start
        print("send takes", t_elapsed)
        if (1.0/FRAMERATE-t_elapsed) > 0 and expected_trasmit_frame == curr_f_id:
            print("capture finished, sleep %f" % (1.0/FRAMERATE-t_elapsed))
            time_to_next_ready_frame = (time.time() - start_timestamp) % (1.0/FRAMERATE)
            time.sleep(1.0/FRAMERATE-time_to_next_ready_frame)


def broadcast_packets(packets, soc, frame_id):
    for packet in packets:
        soc.sendto(packet, ("10.255.255.255", 8888))
    soc.sendto(int.to_bytes(frame_id, 2, 'big'), ("10.255.255.255", 8888))


def parse_v_id_from_addr(ip_addr):
    return int(ip_addr.split('.')[-1]) - 2


def sensor_data_recv(soc):
    while True:
        packet, addr = soc.recvfrom(4096)
        if addr[0] != self_ip:
            v_id = parse_v_id_from_addr(addr[0])
            current_timestamp = time.time()
            connected_vehicle_lock.acquire()
            current_connected_vehicles[v_id] = current_timestamp
            if len(packet) < 56:
                # FIN for a frame
                frame_id = int.from_bytes(packet, 'big')
                print("[V2V Recv frame] from {} {} {}".format(v_id, frame_id, current_timestamp))
                if frame_id not in fully_finished_frame:
                    fully_finished_frame[frame_id] = [v_id]
                else:
                    fully_finished_frame[frame_id].append(v_id)
                if len(fully_finished_frame[frame_id]) == len(current_connected_vehicles.keys()):
                    print("[All possible frame recved] {} {}".format(frame_id, current_timestamp))
            else:
                # print("[chunk recved from peer] {} {}".format(v_id, time.time()))
                OcTree.decode_partial(packet)
            connected_vehicle_lock.release()


def check_inrange_vehicles():
    while True:
        connected_vehicle_lock.acquire()
        current_timestamp = time.time()
        ids_to_pop = []
        for v_id, last_recv_ts in current_connected_vehicles.items():
            if current_timestamp - last_recv_ts > NODE_LEFT_TIMEOUT:
                ids_to_pop.append(v_id)
        for v_id_to_pop in ids_to_pop:
            current_connected_vehicles.pop(v_id_to_pop)
        connected_vehicle_lock.release()
        time.sleep(0.1)


def get_frame_ready_timestamp(frame_id, fps):
    return start_timestamp + frame_id * (1/fps)


def get_curr_tranmist_frame_id():
    return int((time.time() - start_timestamp)*FRAMERATE)


def carspeak_sensor_data_capture(pcd_data_path, oxts_data_path, fps):
    for i in range(config.MAX_FRAMES):
        if pcd_data_type == "GTA":
            pcd_f_name = pcd_data_path + "%06d.bin"%i
            oxts_f_name = oxts_data_path + "%06d.txt"%i
        elif pcd_data_type == "Carla":
            pcd_f_name = pcd_data_path + str(800+i) + ".npy"
            oxts_f_name = oxts_data_path + str(800+i) + ".trans.npy"
        oxts_data_buffer.append(ptcl.pointcloud.read_oxts(oxts_f_name, pcd_data_type))
        pcd_np = ptcl.pointcloud.read_pointcloud(pcd_f_name, pcd_data_type)
        encoded_data = OcTree.loss_resillient_encode(pcd_np)
        pcd_data_buffer.append(encoded_data)


def main():
    global start_timestamp
    while time.time() < args.start_timestamp:
        time.sleep(0.005)
    t_start = time.time()
    print('starting', time.time())
    carspeak_sensor_data_capture(PCD_DATA_PATH, OXTS_DATA_PATH, FRAMERATE)
    t_elapsed = time.time() - t_start
    print("read and encode takes %f" % t_elapsed)
    if t_elapsed < 55:
        time.sleep(55-t_elapsed)
    start_timestamp = time.time()
    print("[start timestamp] ", start_timestamp)
    loction_update_thread = threading.Thread(target=self_loc_update_thread, args=())
    loction_update_thread.daemon = True
    loction_update_thread.start()
    
    vehicle_inrange_t = threading.Thread(target=check_inrange_vehicles, args=())
    vehicle_inrange_t.daemon = True
    vehicle_inrange_t.start()
    
    broadcast_sensor_data()


if __name__ == '__main__':
    main()