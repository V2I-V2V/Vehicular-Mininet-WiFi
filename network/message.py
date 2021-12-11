import time
import struct
import pickle
from tokenize import group

TYPE_CONTROL_MSG = 0
TYPE_DATA_MSG = 1
TYPE_SERVER_REPLY_MSG = 2
TYPE_SEVER_ACK_MSG = 3
TYPE_GROUP = 4
TYPE_LOCATION = 0
TYPE_ASSIGNMENT = 1
TYPE_ROUTE = 2
TYPE_SOS = 3
TYPE_SOS_REPLY = 4
TYPE_FALLBACK = 5
TYPE_RECONNECT = 6
CONTROL_MSG_HEADER_LEN = 6
DATA_MSG_HEADER_LEN = 36
SERVER_REPLY_MSG_HEADER_LEN = 16
MAX_CHUNKS_NUM = 4

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


def construct_data_msg_header(msg_payload, msg_type, frame_id, vehicle_id, frame_ready_timestamp, num_chucks=1,\
    chunk=None):
    """Construct a data message header
    |---- 4 bytes ----|---- 2 bytes ----|---- 2 bytes ----|---- 2 bytes ----|---- 8 bytes ----| ---2 bytes--- |---- 4 bytes ----|---- 4 bytes ----|---- 4 bytes ----| ---- 4 bytes ----|
    | payload length  |    frame id     |    vehicle id   |  message type   |    timestamp    | num of chunks | chunk 1 size | chunk 2 size | chunk 3 size | chunk 4 size |
 
    Args:
        msg_payload (int): length of the msg payload
        msg_type (int): type of msg
        frame_id (int): frame id
        vehicle_id (int): vehicle id
        num_chucks (int): num of chunks
        chunk_sizes (list): list of chunks

    Returns:
        headers [bytes]
    """
    msg_len = len(msg_payload)
    encoded_ts = struct.pack('!d', time.time())
    encoded_frame_ready_time =  struct.pack('!d', frame_ready_timestamp)
    
    header = msg_len.to_bytes(4, "big") + frame_id.to_bytes(2, "big") \
                + vehicle_id.to_bytes(2, "big") + msg_type.to_bytes(2, 'big') \
                + encoded_frame_ready_time + num_chucks.to_bytes(2, "big")
        
    for chunk_i in range(MAX_CHUNKS_NUM):
        base_size = 0
        if type(chunk) is list and chunk_i < len(chunk): # num_chunks is not 1
            base_size = len(chunk[chunk_i])
            print("chunk %d size %d"%(chunk_i, base_size))
        header += base_size.to_bytes(4, 'big')
        
    return header


def construct_reply_msg_header(msg_payload, msg_type, frame_id):
    """Construct server reply message header
    |---- 4 bytes ----|---- 2 bytes ----|---- 2 bytes ----|---- 8 bytes ----|
    | payload length  |    frame id     |  message type   |    timestamp    |

    Args:
        msg_payload ([type]): [description]
        msg_type ([type]): [description]
        frame_id ([type]): [description]
    """
    msg_len = len(msg_payload)
    encoded_ts = struct.pack('!d', time.time())
    header = msg_len.to_bytes(4, 'big') + frame_id.to_bytes(2, 'big') \
                + msg_type.to_bytes(2, 'big') + encoded_ts
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
                # if bytes_sent == 0:
                #     raise RuntimeError("socket connection broken")
            except:
                print('[Send error] connection broken')


def recv_msg(socket, type, is_udp=False):
    if is_udp:
        packet, addr = socket.recvfrom(1024)
        return packet, addr
    else:
        try:
            if type == TYPE_DATA_MSG:
                header_len = DATA_MSG_HEADER_LEN
            elif type == TYPE_CONTROL_MSG:
                header_len = CONTROL_MSG_HEADER_LEN
            elif type == TYPE_SERVER_REPLY_MSG:
                header_len = SERVER_REPLY_MSG_HEADER_LEN
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
            elapsed_time = (time.time() - start_time)
            if type == TYPE_DATA_MSG:
                # print("bytes account %d"%thrpt_cnt_bytes)
                if thrpt_cnt_bytes > 0:
                    throughput = thrpt_cnt_bytes*8.0/1000000/elapsed_time
                else:
                    throughput = msg_payload_size*8.0/1000000/elapsed_time
                return header, payload, throughput, elapsed_time
            else:
                return header, payload
        except:
            print('exception in receiving')
            if type == TYPE_DATA_MSG:
                return b'', b'', 0.0, 0.0
            else:
                return b'', b''


def parse_server_reply_msg_header(data):
    payload_size = int.from_bytes(data[0:4], 'big')
    frame_id = int.from_bytes(data[4:6], "big")
    msg_type = int.from_bytes(data[6:8], 'big')
    ts = struct.unpack('!d', data[8:16])[0]
    return payload_size, frame_id, msg_type, ts


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
    num_chunks = int.from_bytes(data[18:20], "big")
    chunk_sizes = []
    for i in range(MAX_CHUNKS_NUM):
        chunk_sizes.append(int.from_bytes(data[20+i*4:20+(i+1)*4], "big"))

    return payload_size, frame_id, v_id, type, ts, num_chunks, chunk_sizes


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
    # x = int.from_bytes(data[2:4], "big")
    x = struct.unpack('!d', data[2:10])[0]
    # y = int.from_bytes(data[4:6], "big")
    y = struct.unpack('!d', data[10:18])[0]
    seq_num = int.from_bytes(data[18:22], "big")
    group_id = pickle.loads(data[22:])
    return helpee_id, [x, y], seq_num, group_id


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
    seq_num = int.from_bytes(data[-12:-8], "big")
    group_id = (int.from_bytes(data[-8:-4], "big"), int.from_bytes(data[-4:], "big"))
    return helpee_id, route_bytes, seq_num, group_id


def vehicle_parse_sos_packet_data(data):
    vehicle_id = int.from_bytes(data[0:2], 'big')
    return vehicle_id


def server_parse_location_msg(msg_payload):
    v_type = int.from_bytes(msg_payload[0:2], "big")
    v_id = int.from_bytes(msg_payload[2:4], "big")
    # x = int.from_bytes(msg_payload[4:6], "big")
    x = struct.unpack('!d', msg_payload[4:12])[0]
    # y = int.from_bytes(msg_payload[6:8], "big")
    y = struct.unpack('!d', msg_payload[12:20])[0]
    seq_num = int.from_bytes(msg_payload[20:24], "big")
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
