# Vehicular perception server

import scheduling
import socket
import threading
import time

location_map = {}
client_sockets = {}
vehicle_types = {} # 0 for helpee, 1 for helper


class SchedThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)


    def run(self):
        while True:
            print(location_map)
            if len(location_map) == 6:
                positions = []
                for k, v in sorted(location_map.items()):
                    positions.append(v)
                helpee_count = 0
                helper_count = 0
                for k, v in vehicle_types.items():
                    if v == 0:
                        helpee_count += 1
                    else:
                        helper_count += 1

                assignment = scheduling.min_total_distance_sched(helpee_count, helper_count, positions)
                print(assignment)
                # for node_num in range(len(positions)):
                #     if node_num in assignment:
                for cnt, node in enumerate(assignment):
                    print("send %d to node %d" % (cnt, node))
                    # print(cnt, node, client_sockets[node])
                    msg = cnt.to_bytes(2, 'big')
                    client_sockets[node].send(msg)
                for node_num in range(0, helpee_count+helper_count):
                    if node_num not in assignment:
                        print("send %d to node %d" % (65535, node_num))
                        msg = int(65535).to_bytes(2, 'big')
                        client_sockets[node_num].send(msg)
            time.sleep(1)


class ConnectionThread(threading.Thread):

    def __init__(self, client_address, client_socket):
        threading.Thread.__init__(self)
        self.client_socket = client_socket
        self.client_address = client_address
        print("New connection added: ", client_address)


    def run(self):
        print("Connection from : ", self.client_address)
        data = self.client_socket.recv(2)
        vehicle_id = int.from_bytes(data, "big")
        client_sockets[vehicle_id] = self.client_socket
        data = self.client_socket.recv(8)
        while data:
            v_type = int.from_bytes(data[0:2], "big")
            v_id = int.from_bytes(data[2:4], "big")
            x = int.from_bytes(data[4:6], "big")
            y = int.from_bytes(data[6:8], "big")
            location_map[v_id] = (x, y)
            vehicle_types[v_id] = v_type
            data = self.client_socket.recv(8)
            # if location map is already constructed, we can do a schedule instantly
            # if len(location_map) == 8:
            #     positions = []
            #     for k, v in sorted(location_map.items()):
            #         positions.append(v)
            #     helpee_count = 0
            #     helper_count = 0
            #     for k, v in vehicle_types.items():
            #         if v == 0:
            #             helpee_count += 1
            #         else:
            #             helper_count += 1

            #     assignment = scheduling.min_total_distance_sched(helpee_count, helper_count, positions)
            #     print(assignment)
            #     # for node_num in range(len(positions)):
            #     #     if node_num in assignment:
            #     for cnt, node in enumerate(assignment):
            #         print("send %d to node %d" % (cnt, node))
            #         # print(cnt, node, client_sockets[node])
            #         msg = cnt.to_bytes(2, 'big')
            #         client_sockets[node].send(msg)
            #     for node_num in range(0, helpee_count+helper_count):
            #         if node_num not in assignment:
            #             print("send %d to node %d" % (65535, node_num))
            #             msg = int(65535).to_bytes(2, 'big')
            #             client_sockets[node_num].send(msg)
        self.client_socket.close()


def main():
    HOST = ''
    PORT = 6666
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    print("Vehicular perception server started")
    print("Waiting for client request..")
    sched_thread = SchedThread()
    sched_thread.daemon = True
    sched_thread.start()
    while True:
        server.listen(1)
        client_socket, client_address = server.accept()
        newthread = ConnectionThread(client_address, client_socket)
        newthread.daemon = True
        newthread.start()


if __name__ == "__main__":
    main()
