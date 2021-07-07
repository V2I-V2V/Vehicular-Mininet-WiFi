TYPE_CONTROL_MSG = 0
TYPE_DATA_MSG = 1
TYPE_LOCATION = 0
TYPE_ASSIGNMENT = 1
TYPE_ROUTE = 2
CONTROL_MSG_HEADER_LEN = 6
DATA_MSG_HEADER_LEN = 10

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
    msg_len = len(msg_payload)
    header = msg_len.to_bytes(4, "big") + frame_id.to_bytes(2, "big") \
                + vehicle_id.to_bytes(2, "big") + msg_type.to_bytes(2, 'big')
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
                print("[Socket closed]")
                return b'', b''
            header_to_recv -= len(data_recv)
        msg_payload_size = int.from_bytes(header[0:4], "big")
        payload = b''
        to_recv = msg_payload_size
        while len(payload) < msg_payload_size:
            data_recv = socket.recv(65536 if to_recv > 65536 else to_recv)
            if len(data_recv) <= 0:
                print("[Socket closed]")
                return b'', b''
            payload += data_recv
            to_recv = msg_payload_size - len(payload)
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
    return payload_size, frame_id, v_id, type


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
