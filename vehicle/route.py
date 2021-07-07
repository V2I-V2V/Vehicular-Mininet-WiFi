import message
import pyroute2
import socket

def get_routes(vehicle_id):
    routing_table = {}
    routes = pyroute2.IPRoute().get_routes()
    for route in routes:
        print(route)
        # print(route['attrs'])
        dst = ""
        nexthop = ""
        for attr in route['attrs']:
            if 'DST' in attr[0]:
                print('DST', attr[1])
                dst = attr[1]
            elif 'GATEWAY' in attr[0]:
                print('GATEWAY', attr[1])
                nexthop = attr[1]
        dst_ip = dst.split('.')
        if dst_ip[0:3] == ['10', '0', '0'] and int(dst_ip[3]) >= 2 and int(dst_ip[3]) != vehicle_id + 2:
            routing_table[dst_ip[-1]] = nexthop.split('.')[-1]
        else:
            print(dst_ip[0:3])
    print(routing_table)
    return routing_table


def broadcast_route(vehicle_id, routing_table, source_socket, seq_num):
    msg = vehicle_id.to_bytes(2, 'big')
    for k, v in routing_table.items():
        msg += int(k).to_bytes(1, 'big') + int(v).to_bytes(1, 'big')
    msg += seq_num.to_bytes(4, 'big')
    header = message.construct_control_msg_header(msg, message.TYPE_ROUTE)
    message.send_msg(source_socket, header, msg, is_udp=True,\
                        remote_addr=("10.255.255.255", 8888))


if __name__ == "__main__":
    routing_table = get_routes(0)
    # Use UDP socket for broadcasting
    v2v_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, \
                                                    socket.IPPROTO_UDP)
    v2v_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    v2v_control_socket.bind(('', 8888))
    broadcast_route(0, routing_table, v2v_control_socket)
