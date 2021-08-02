import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import network.message
import pyroute2
import socket

def get_routes(vehicle_id):
    routing_table = {}
    routes = pyroute2.IPRoute().get_routes()
    for route in routes:
        # print(route)
        # print(route['attrs'])
        dst = ""
        nexthop = ""
        for attr in route['attrs']:
            if 'DST' in attr[0]:
                # print('DST', attr[1])
                dst = attr[1]
            elif 'GATEWAY' in attr[0]:
                # print('GATEWAY', attr[1])
                nexthop = attr[1]
            elif 'MULTIPATH' in attr[0]:
                # print('MULTIPATH', attr[1])
                nexthop = attr[1][0]['attrs'][0][1]
                # print('NEXTHOP', nexthop)

        dst_ip = dst.split('.')
        if dst_ip[0:3] == ['10', '0', '0'] and int(dst_ip[3]) >= 2 and int(dst_ip[3]) != vehicle_id + 2:
            # vehicle_id = IP - 2
            # print('dst_ip %s, nexthop: %s'%(dst_ip, nexthop))
            # if nexthop != "":
            routing_table[int(dst_ip[-1]) - 2] = int(nexthop.split('.')[-1]) - 2
        else:
            # print(dst_ip[0:3])
            pass
    # print(routing_table)
    return routing_table


def table_to_bytes(routing_table):
    route_bytes = b''
    for k, v in routing_table.items():
        route_bytes += int(k).to_bytes(1, 'big') + int(v).to_bytes(1, 'big')
    return route_bytes


def broadcast_route(vehicle_id, routing_table, source_socket, seq_num):
    msg = vehicle_id.to_bytes(2, 'big') + table_to_bytes(routing_table) + seq_num.to_bytes(4, 'big')
    header = network.message.construct_control_msg_header(msg, network.message.TYPE_ROUTE)
    network.message.send_msg(source_socket, header, msg, is_udp=True,\
                        remote_addr=("10.255.255.255", 8888))


def get_routing_path(helpee, helper, routing_tables):
    # find the path from a helpee to a helper
    i = helpee
    routing_path = [i]
    while i != helper:
        if i not in routing_tables.keys() or helper not in routing_tables[i].keys():
            # cannot reach destination, return an empty array
            return []
        i = routing_tables[i][helper]
        routing_path.append(i)
    return routing_path


def get_num_hops(helpee, helper, routing_tables):
    # find the number of hops from a helpee to a helper
    if len(get_routing_path(helpee, helper, routing_tables)) == 0:
        return 65535 # a large number
    else:
        return len(get_routing_path(helpee, helper, routing_tables)) - 1


def get_neighbors(node, routing_tables):
    neighbors = []
    if node in routing_tables.keys():
        for k, v in routing_tables[node].items():
            if k == v:
                neighbors.append(k)
    return neighbors


if __name__ == "__main__":
    routing_table = get_routes(0)
    # routes = pyroute2.IPRoute().get_routes()
    print(routing_table)
    # routing_tables = {4: {0: 2, 1: 2, 2: 2, 3: 3, 5: 5}, 2: {0: 0, 1: 1, 3: 3, 4: 4, 5: 5}, 
    #                   5: {0: 2, 1: 2, 2: 2, 3: 3, 4: 4}, 3: {0: 0, 1: 1, 2: 2, 4: 4, 5: 5}, 
    #                   1: {0: 0, 2: 2, 3: 3, 4: 2, 5: 2}, 0: {1: 1, 2: 2, 3: 3, 4: 2, 5: 2}}
    # routing_path = get_routing_path(0, 5, routing_tables)
    # num_hops = get_num_hops(0, 5, routing_tables)
    # print(routing_path, num_hops)
