# this file handles throughput calculation, signal strength info related calculation
import numpy as np
# import config
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
    disconnect_dict = {0: [2.1], 1: [2.12]}
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