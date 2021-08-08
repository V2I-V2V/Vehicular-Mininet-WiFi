import numpy as np

MAX_FRAMES = 80

simulation_time = 10.0
server_ip = "192.168.0.1"
server_ctrl_port = 6666
server_data_port = 6667

map_scheduler_to_int_encoding = {'combined': 0, 'minDist': 1, 'routeAware': 2, 'bwAware':3, 'random': 4, 'fixed': 5, 'distributed': 6}
map_int_encoding_to_scheduler = {0: 'combined', 1: 'minDist', 2: 'routeAware', 3: 'bwAware', 4: 'random', 
                                5: 'fixed', 6: 'distributed'}

# global wwan_sockets
# wwan_sockets = []
# global adhoc_broadcast_sinks
# adhoc_broadcast_sinks = []

# global adhoc_broadcast_source
# adhoc_broadcast_source = []

# global ipAddr
# ipAddr = [] # change to ns3.ipv4container
# global locations
# locations = []
# global source_sock
# source_sock = None

# global current_assigned_helpee
# current_assigned_helpee = np.zeros(8)
# current_assigned_helpee[:] = 65535

# global vehicular_app_list
# vehicular_app_list = []

# global is_helper_list
# is_helper_list = np.ones(8)
