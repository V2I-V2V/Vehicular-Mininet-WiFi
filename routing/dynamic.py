import os, sys
import socket
import time
import threading
import argparse
import networkx as nx
import json

def dict_to_binary(the_dict):
    str = json.dumps(the_dict)
    binary = ' '.join(format(ord(letter), 'b') for letter in str)
    return binary


def binary_to_dict(the_binary):
    jsn = ''.join(chr(int(x, 2)) for x in the_binary.split())
    d = json.loads(jsn)  
    return d

ars = argparse.ArgumentParser()

# LSA, LSDB

ROUTE_NUM = 2
BROADCAST_INTERVAL = 0.5
VALID_INTERVAL = 2.0
seq_num = 0
seq_num_dict = {}
self_id = int(sys.argv[1])
self_ip = "10.0.0." + str(self_id+2)
self_ifname = 'sta'+str(self_id)+'-wlan0'
node_neighbour_lock = threading.Lock()
node_neighbours = {self_id: []}
route_established = {}
last_neighbour_recv_ts = {}
node_neighbours_update = False

broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
host_ip = ''
host_port = 5558
broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
broadcast_sock.bind((host_ip, host_port))


def node_ip(node_num):
    return "10.0.0." + str(node_num+2)

def update_routing_thread():
    global node_neighbours_update
    while True:
        if node_neighbours_update:
            print("update routing")
            G = nx.Graph()
            node_neighbour_lock.acquire()
            print(node_neighbours)
            all_nodes = node_neighbours.keys()
            G.add_nodes_from(node_neighbours.keys())
            for k, v in node_neighbours.items():
                for v in node_neighbours[k]:
                    G.add_edge(k, v)
            for n_id in all_nodes:
                if n_id != self_id:
                    try:
                        X = nx.shortest_simple_paths(G, self_id, n_id)
                        print("path from %d to %d" % (self_id, n_id))
                        cnt = 0
                        next_hops = []
                        for counter, path in enumerate(X):
                            print(path)
                            next_hops.append(path[1])
                            cnt += 1
                            if counter == ROUTE_NUM-1:
                                break
                        # multipath routing
                        print("path cnt: %d" % cnt)
                        print(next_hops)
                        if cnt == ROUTE_NUM:
                            print("configure multipath %s, %s, %s" % (node_ip(n_id), node_ip(next_hops[0]), node_ip(next_hops[1])))
                            os.system('ip route replace %s nexthop via %s weight 100 nexthop via %s weight 1'\
                                % (node_ip(n_id), node_ip(next_hops[0]), node_ip(next_hops[1])))
                        elif cnt == 1:
                            print("configure singlepath")
                            os.system('ip route replace %s nexthop via %s' % (node_ip(n_id), node_ip(next_hops[0])))
                    except Exception as error:
                        print(error)
                        pass
            node_neighbours_update = False
            node_neighbour_lock.release()   
        time.sleep(0.5)


def send_neighbour_info():
    global seq_num
    msg = seq_num.to_bytes(4, 'big')
    msg += self_id.to_bytes(1, 'big')
    msg += dict_to_binary(node_neighbours).encode('utf-8')
    print("broadcast from %d" % self_id)
    broadcast_sock.sendto(msg, ("10.255.255.255", 5558))
    seq_num += 1

def periodic_broadcast_neighbour():
    while True:
        send_neighbour_info()
        time.sleep(1)


def add_node_to_node_neighbours(node_id):
    if self_id not in node_neighbours.keys():
        node_neighbours[self_id] = [node_id]
    elif node_id not in node_neighbours[self_id]:
        node_neighbours[self_id].append(node_id)


def check_if_seq_newer(seq, node_id):
    return (node_id not in seq_num_dict.keys() or seq > seq_num_dict[node_id])

def parse_neighbour_info(data, addr):
    global node_neighbours_update
    seq = int.from_bytes(data[0:4], 'big')
    next_hop = int.from_bytes(data[4:5], 'big')
    rebroadcast = False
    if next_hop != self_id:
        if check_if_seq_newer(seq, next_hop):
            seq_num_dict[next_hop] = seq
            rebroadcast = True
        print("recv broadcast from %d %s" % (next_hop, addr[0]))
        next_hop_ip = node_ip(next_hop)
        neighbour_dict = binary_to_dict(data[5:].decode('utf-8'))
        print(neighbour_dict)
        node_neighbour_lock.acquire()
        if next_hop_ip == addr[0]:
            print("update neighbour timestamp")
            add_node_to_node_neighbours(next_hop)
            last_neighbour_recv_ts[next_hop] = time.time()
        node_neighbours[next_hop] = neighbour_dict[str(next_hop)]
        node_neighbours_update = True
        node_neighbour_lock.release()  
    return rebroadcast
     

def clear_outdated_neighbour_info():
    while True:
        node_neighbour_lock.acquire()
        curr_time = time.time()
        for n, t in last_neighbour_recv_ts.items():
            if curr_time - t > VALID_INTERVAL:
                print("%d outdated %f" % (n, time.time()))
                node_neighbours[n] = []
                # for node_id in node_neighbours.keys():   
                if n in node_neighbours[self_id]:
                    node_neighbours[self_id].remove(n)
        node_neighbour_lock.release()
        time.sleep(BROADCAST_INTERVAL)
            

def recv_neighbour_broadcast():
    while True:
        data, addr = broadcast_sock.recvfrom(1024)
        if addr[0] != self_ip:
            rebroadcast = parse_neighbour_info(data, addr)
            if rebroadcast:
                broadcast_sock.sendto(data, ("10.255.255.255", 5558))


def main():
    recv_thread = threading.Thread(target=recv_neighbour_broadcast, args=())
    # send_thread = threading.Thread(target=send_neighbour_info, args=())
    recv_thread.start()
    # send_thread.start()
    route_update_thread = threading.Thread(target=update_routing_thread, args=())
    route_update_thread.start()
    clear_stale_neighbour_info_thread = threading.Thread(target=clear_outdated_neighbour_info, args=())
    clear_stale_neighbour_info_thread.start()
    periodic_broadcast_neighbour()

if __name__ == '__main__':
    main()
