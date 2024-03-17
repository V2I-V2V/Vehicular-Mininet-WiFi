import numpy as np

MAX_FRAMES = 80

simulation_time = 10.0
server_ip = "192.168.0.1"
server_ctrl_port = 6666
server_data_port = 6667

map_scheduler_to_int_encoding = {'combined': 0, 'minDist': 1, 'routeAware': 2, 'bwAware':3, 'random': 4, 'fixed': 5, 
                                 'distributed': 6, 'v2i': 7, 'v2v': 8}
map_int_encoding_to_scheduler = {0: 'combined', 1: 'minDist', 2: 'routeAware', 3: 'bwAware', 4: 'random', 5: 'fixed', 
                                 6: 'distributed', 7: 'v2i', 8: 'v2v'}

