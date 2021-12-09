import network.message as message
def broadcast_sos_msg(socket, vehicle_id, self_loc=None):
    # broadcast sos msg from helpee nodes
    msg_payload = vehicle_id.to_bytes(2, 'big')
    header = message.construct_control_msg_header(msg_payload, message.TYPE_SOS)
    message.send_msg(socket, header, msg_payload, is_udp=True, \
        remote_addr=("10.0.0.255", 8888))

def echo_sos_msg(socket, vehicle_id, dest):
    # reply to a sos msg that I can help you
    msg_payload = vehicle_id.to_bytes(2, 'big')
    header = message.construct_control_msg_header(msg_payload, message.TYPE_SOS_REPLY)
    message.send_msg(socket, header, msg_payload, is_udp=True, \
        remote_addr=dest)