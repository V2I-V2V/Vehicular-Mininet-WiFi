import pickle
import network.message as message

def find_vehicle_location_in_group(group_id, group_id_to_vehicles, location_map):
    """Given a group id, return a map consisting of vehicles' location in the group
    """
    location_map_of_group = {}
    if group_id in group_id_to_vehicles:
        vehicles_in_group = group_id_to_vehicles[group_id]
        for v_id in vehicles_in_group:
            location_map_of_group[v_id] = location_map[v_id]
    return location_map_of_group


def find_vehicle_route_in_group(group_id, group_id_to_vehicles, route_map):
    """Given a group id, return a map consisting of vehicles' routing paths in the group
    """
    route_map_of_group = {}
    if group_id in group_id_to_vehicles:
        vehicles_in_group = group_id_to_vehicles[group_id]
        for v_id in vehicles_in_group:
            route_map_of_group[v_id] = route_map[v_id] # Do we need to remove other routes in this case?
    return route_map_of_group   


def devide_vehicle_to_groups(location_map):
    """Devide vehicle into different groups based on their location, return a dictionary with group id as key
    and vehicle ids as value
    """
    group_id_to_vehicles = {}
    for vehicle_id, vehicle_loc in location_map.items():
        group = get_group_id_based_on_location(vehicle_loc)
        if group not in group_id_to_vehicles:
            group_id_to_vehicles[group] = [vehicle_id]
        else:
            group_id_to_vehicles[group].append(vehicle_id)
    return group_id_to_vehicles


# recursively partition the group in smaller sizes (containing less vehicles) if # of vehicles 
# is more than a threshold x
def get_group_id_based_on_location(location, grid_size=100):
    x_id, y_id = int(location[0]/grid_size), int(location[1]/grid_size)
    return (x_id, y_id)


def notify_group_change(client_sockets, group_map):
    for g_id, v_ids in group_map.items():
        # send g_id to v_ids
        payload = pickle.dumps(g_id)
        header = message.construct_control_msg_header(payload, message.TYPE_GROUP)
        for v_id in v_ids:
            message.send_msg(client_sockets[v_id], header, payload)


