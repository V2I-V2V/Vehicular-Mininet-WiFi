# Control the locations of vehicles
import numpy as np
import network.message
import time
import mobility_noise
import pickle
import struct

def read_locations(location_filename):
    # TODO
    # Read a mobility trace which has m column and n lines
    # m is 2 * number of vehicles (x1, y1, x2, y2 ...)
    # n is total number of locations that need to update at 100ms granularity
    trace = np.loadtxt(location_filename)
    if trace.ndim == 1:
        trace = trace.reshape(1, -1)
    trace_dict = {}
    total_node_num = int(trace.shape[1]/2)
    for i in range(total_node_num):
        node_trace = trace[:,2*i:2*i+2]
        trace_dict[i] = node_trace
    print("read locations from files")
    return trace_dict

def init_location(location, mobility_model=None):
    """Define a initialization function for location, beacuse the update will have different API

    Args:
        location (np array/list of float): [x,y] of the node location
        mobility_model (str): mobility model
    """
    print("Initialize location")

def update_location(node, locations):
    """Update location

    Args:
        node: node to update position
        location (list): list of locations of this node (mobility trace)
    """
    print("update location")



def broadcast_location(vehicle_id, self_loc, source_socket, seq_num, group_id, add_noise=False):
    if add_noise:
        x,y = mobility_noise.add_random_noise_on_loc(self_loc[0], self_loc[1], std_deviation=30.0)
    else:
        x, y = self_loc[0], self_loc[1]
    msg = vehicle_id.to_bytes(2, 'big') +  struct.pack('!d', x) \
        + struct.pack('!d', y) + seq_num.to_bytes(4, 'big')
    msg += pickle.dumps(group_id)
    header = network.message.construct_control_msg_header(msg, network.message.TYPE_LOCATION)
    # print("[Loc msg size] %d %f"%(len(msg)+len(header), time.time()))
    network.message.send_msg(source_socket, header, msg, is_udp=True,\
                        remote_addr=("10.0.0.255", 8888))
    # source_socket.sendto(msg, ("10.255.255.255", 8888))

        
def stop_broadcast_location(node, source_socket):
    pass

def advance_loc_positive_x(loc, node_num):
    loc[node_num*2] += 1.5
    return loc

def advance_loc_negative_x(loc, node_num):
    loc[node_num*2] -= 1.5
    return loc


def generate_mobility_traces():
    start_loc = np.array([24.49976316285629, 217.71708884851688,  357.74520954696743,  167.77189417656552,  79.35720843074337,  344.36319984978942,  247.61379340033827,  68.87204405592615,  350.67895221295004,  341.09553616202544,  339.9494297000563,  23.174821747631213,  259.94083181758583,  158.5891261090795,  12.755221427841423,  144.30321364694115])
    print(start_loc)
    result_trace = start_loc
    for i in range(99):
        loc = advance_loc_positive_x(start_loc, 0)
        loc = advance_loc_negative_x(loc, 6)
        print(start_loc)
        result_trace = np.vstack((result_trace, loc))
    np.savetxt('mob_trace.txt', result_trace, delimiter=' ') 


def main():
    generate_mobility_traces()


if __name__ == "__main__":
    main()

