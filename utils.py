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
    disconnect_dict = {0: [2.1], 2: [2.12]}
    print("Process traces and get the disconneted timestamps of each node")
    return disconnect_dict