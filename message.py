import time
import struct

TYPE_CONTROL_MSG = 0
TYPE_DATA_MSG = 1
TYPE_LOCATION = 0
TYPE_ASSIGNMENT = 1
TYPE_ROUTE = 2
CONTROL_MSG_HEADER_LEN = 6
DATA_MSG_HEADER_LEN = 18

def construct_control_msg_header(msg_payload, msg_type):
    """ Construct a control message header
    |---- 4 bytes ----|---- 2 bytes ----|
    | message length  |  message type   |   

    Args:
        msg_payload (bytes): message payload to send (in raw bytes)
        msg_type (int): type of control messages, can be 0 (location), 1 (assignment), 2 (routing)

    Returns:
        headers [bytes]
    """
    msg_len = len(msg_payload)
    header = msg_len.to_bytes(4, 'big') + msg_type.to_bytes(2, 'big')
    return header


def construct_data_msg_header(msg_payload, msg_type, frame_id, vehicle_id):
    """Construct a data message header
    |---- 4 bytes ----|---- 2 bytes ----|---- 2 bytes ----|---- 2 bytes ----|---- 8 bytes ----|
    | message length  |    frame id     |    vehicle id   |  message type   |    timestamp    |
 
    Args:
        msg_payload (int): length of the msg payload
        msg_type (int): type of msg
        frame_id (int): frame id
        vehicle_id (int): vehicle id

    Returns:
        headers [bytes]
    """
    msg_len = len(msg_payload)
    encoded_ts = struct.pack('!d', time.time())
    
    header = msg_len.to_bytes(4, "big") + frame_id.to_bytes(2, "big") \
                + vehicle_id.to_bytes(2, "big") + msg_type.to_bytes(2, 'big') \
                + encoded_ts
    return header


def send_msg(socket, header, msg_payload, is_udp=False, remote_addr=None):
    """ General method to send control/data messages
    """
    if is_udp:
        # UDP must sent header and payload in one DGRAM
        packet = header + msg_payload
        socket.sendto(packet, remote_addr)
    else:
        # send msg header
        hender_sent = 0
        while hender_sent < len(header):
            bytes_sent = socket.send(header[hender_sent:])
            hender_sent += bytes_sent

        # send msg payload
        total_sent = 0
        msg_len = len(msg_payload)
        while total_sent < msg_len:
            try:
                bytes_sent = socket.send(msg_payload[total_sent:])
                total_sent += bytes_sent
                if bytes_sent == 0:
                    raise RuntimeError("socket connection broken")
            except:
                print('[Send error] connection broken')


def recv_msg(socket, type, is_udp=False):
    if is_udp:
        packet, addr = socket.recvfrom(1024)
        return packet, addr
    else:
        if type == TYPE_DATA_MSG:
            header_len = DATA_MSG_HEADER_LEN
        elif type == TYPE_CONTROL_MSG:
            header_len = CONTROL_MSG_HEADER_LEN
        header = b''
        header_to_recv = header_len
        while len(header) < header_len:
            data_recv = socket.recv(header_to_recv)
            header += data_recv
            if len(data_recv) <= 0:
                if type == TYPE_DATA_MSG:
                    return b'', b'', 0.0, 0.0
                else:
                    return b'', b''
            header_to_recv -= len(data_recv)
        msg_payload_size = int.from_bytes(header[0:4], "big")
        payload = b''
        to_recv = msg_payload_size
        start_time = time.time()
        #### TODO recv 4000 each time, discard interval and data with 0.0s,
        #### calculate throughput based on first chunk that >0.0s
        buffer_drained = False
        thrpt_cnt_bytes = 0
        first_buffer_empty_recv_time = 0.0
        while len(payload) < msg_payload_size:
            # recv_s = time.time()
            data_recv = socket.recv(65536 if to_recv > 65536 else to_recv)
            # recv_elapsed = time.time() - recv_s
            # # print("recv take %f"%recv_elapsed)
            # if buffer_drained is False and recv_elapsed > 0.0003:
            #     start_time = time.time()
            #     buffer_drained = True
            #     first_buffer_empty_recv_time = recv_elapsed
            # if buffer_drained:
            #     thrpt_cnt_bytes += len(data_recv)
            if len(data_recv) < 0:
                print("[Socket closed]")
                if type == TYPE_DATA_MSG:
                    return b'', b'', 0.0, 0.0
                else:
                    return b'', b''
            payload += data_recv
            to_recv = msg_payload_size - len(payload)
        elapsed_time = first_buffer_empty_recv_time + (time.time() - start_time)
        if type == TYPE_DATA_MSG:
            # print("bytes account %d"%thrpt_cnt_bytes)
            if thrpt_cnt_bytes > 0:
                throughput = thrpt_cnt_bytes*8.0/1000000/elapsed_time
            else:
                throughput = msg_payload_size*8.0/1000000/elapsed_time
            return header, payload, throughput, elapsed_time
        else:
            return header, payload


def parse_control_msg_header(data):
    payload_size = int.from_bytes(data[0:4], 'big')
    msg_type = int.from_bytes(data[4:6], 'big')
    return payload_size, msg_type


def parse_data_msg_header(data):
    payload_size = int.from_bytes(data[0:4], "big")
    frame_id = int.from_bytes(data[4:6], "big")
    v_id = int.from_bytes(data[6:8], "big")
    type = int.from_bytes(data[8:10], "big")
    ts = struct.unpack('!d', data[10:18])[0]
    return payload_size, frame_id, v_id, type, ts


def vehicle_parse_location_packet_data(data):
    """Parse location packet, packet should be of length 10 = 2 + 2 + 2 + 4

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


def vehicle_parse_route_packet_data(data):
    """Parse route packet, packet should be of length (2 + (1 + 1) * (n_vechiles - 1) + 4)

    Args:
        data (bytes): raw network packet data to parse

    Returns:
        helpee_id: helpee node id that sends the packet
        [x, y]: location of the helpee node
    """
    # return helpee id, route, seq
    l = len(data)
    helpee_id = int.from_bytes(data[0:2], "big")
    route_bytes = data[2:-4]
    seq_num = int.from_bytes(data[-4:], "big")
    return helpee_id, route_bytes, seq_num


def server_parse_location_msg(msg_payload):
    v_type = int.from_bytes(msg_payload[0:2], "big")
    v_id = int.from_bytes(msg_payload[2:4], "big")
    x = int.from_bytes(msg_payload[4:6], "big")
    y = int.from_bytes(msg_payload[6:8], "big")
    seq_num = int.from_bytes(msg_payload[8:12], "big")
    return v_type, v_id, x, y, seq_num


def server_parse_route_msg(msg_payload):
    v_type = int.from_bytes(msg_payload[0:2], "big")
    v_id = int.from_bytes(msg_payload[2:4], "big")
    routing_table = {}
    i = 4
    while i < len(msg_payload) - 4:
        routing_table[int.from_bytes(msg_payload[i:i+1], "big")] = int.from_bytes(msg_payload[i+1:i+2], "big")
        i += 2
    seq_num = int.from_bytes(msg_payload[-4:], "big")
    return v_type, v_id, routing_table, seq_num
