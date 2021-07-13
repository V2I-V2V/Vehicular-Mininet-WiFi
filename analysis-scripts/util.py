import numpy as np

def get_sender_ts(filename):
    sender_ts = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("[V2"):
                sender_ts.append(float(line.split()[-1]))
    return sender_ts


def get_receiver_ts(filename):
    receiver_ts_dict = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("[Full frame recved]"):
                parse = line.split()
                sender_id = int(parse[4][:-1])
                ts = float(parse[-1])
                if sender_id in receiver_ts_dict.keys():
                    receiver_ts_dict[sender_id].append(ts)
                else:
                    receiver_ts_dict[sender_id] = [ts]
        f.close()
    return receiver_ts_dict