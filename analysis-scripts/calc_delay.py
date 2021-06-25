import sys, os
import numpy as np

np.set_printoptions(precision=3)

sender_ts_dict = {}
receiver_ts_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
delay_dict = {}

def get_sender_ts(filename):
    sender_ts = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("[V2"):
                sender_ts.append(float(line.split()[-1]))
    return sender_ts

def get_receiver_ts(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("[Full frame recved]"):
                parse = line.split()
                sender_id = int(parse[4][:-1])
                ts = float(parse[-1])
                receiver_ts_dict[sender_id].append(ts)
    


def main():
    for i in range(6):
        sender_ts_dict[i] = get_sender_ts('../logs/node%d.log'%i)
    # print(sender_ts_dict)
    get_receiver_ts('../logs/server.log')
    # print(len(receiver_ts_dict[0]))
    for i in range(6):
        try:
            delay_dict[i] = np.array(receiver_ts_dict[i]) \
                            - np.array(sender_ts_dict[i])
            print(delay_dict[i])
        except:
            pass

if __name__ == '__main__':
    
    main()
