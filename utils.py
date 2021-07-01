# this file handles throughput calculation, signal strength info related calculation
import numpy as np
import math
import os
import random

def random_number(start, end):
    return (end - start) * random.random() + start

def random_int(start, end):
    return int((end - start) * random.random() + start)


def read_traces(num_nodes):
    traces = []
    print("Read LTE traces")
    return traces


def process_traces(traces):
    disconnect_dict = {0: [2.1], 1: [2.1]}
    # disconnect_dict ={0: [2.1]}
    print("Process traces and get the disconneted timestamps of each node")
    return disconnect_dict


def produce_3d_location_arr(location):
    location_3d_arr = []
    for i in range(int(len(location)/2)):
        location_str = str(location[2*i])+','+str(location[2*i+1])+',0'
        location_3d_arr.append(location_str)
    return location_3d_arr

def produce_assignment_str(assignment):
    assignment_str = ""
    for i in assignment:
        assignment_str = assignment_str + str(i) + ' '
    return assignment_str


def parse_offline_settings(settings):
    """Parse the setting file
    Args:
        settings ([str]): [setting file name]
    Returns:
        total locations
        total node number
        helpee node number
        helper node number
        locations, size (total locations, 2*total node number)
        assignments, with each index containing all assignment of one location
    """
    with open(settings, 'r') as f:
        parse = next(f).split()
        total_loc, num_nodes = int(parse[0]), int(parse[1])
        locations = []
        assignments = []
        for i in range(total_loc):
            location = np.array(next(f).split(), dtype=float)
            # print(location)
            node_info = next(f).split()
            num_helpee, num_helper = int(node_info[0]), int(node_info[1])
            num_assignment_schemes = int(math.factorial(num_helper)/math.factorial(num_helpee-1))
            assignment = np.array([next(f).split() for x in range(num_assignment_schemes)])
            locations.append(location)
            assignments.append(assignment)
            # print(assignment)
    return total_loc, num_nodes, num_helpee, num_helper, locations, assignments
