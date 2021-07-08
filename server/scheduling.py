# Schedule helpers for helpees according to different algorithms
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vehicle.route
import itertools
import math
import random

def find_all_one_to_one(num_of_helpees, num_of_helpers):
    '''
        number_of_helpees smaller than or equal to num_of_helpers
    '''
    helpees = [i for i in range(0, num_of_helpees)]
    helpers = [i + num_of_helpees for i in range(0, num_of_helpers)]
    # print(helpers, helpees)
    assignments = [x for x in itertools.permutations(helpers, len(helpees))]
    # print(len(assignments), assignments)
    return assignments


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_in_range(tx_position, rx_position, tx_coverage):
    '''
        Check if a rx is in the coverage of a tx
    '''
    if get_distance(tx_position, rx_position) < tx_coverage:
        return True
    else:
        return False

def get_assignment_tuple(assignment_list):
    return tuple(assignment_list)

def get_assignment_from_id(assignment_id):
    '''
        Assignment is like (4, 7, 5)
    '''
    assignment = []
    for item in assignment_id.split('-'):
        assignment.append(int(item))
    return tuple(assignment)


def get_id_from_assignment(assignment):
    '''
        Assignment ID is like "4-7-5"
    '''
    assignment_id = ''
    for node in assignment:
        assignment_id += str(node) + "-"
    assignment_id = assignment_id[:-1]
    return assignment_id


def get_nodes_on_routes(assignment, routing_tables):
    nodes_on_routes = set()
    for helpee, helper in enumerate(assignment):
        for node in vehicle.route.get_routing_path(helpee, helper, routing_tables):
            nodes_on_routes.add(node)
    return nodes_on_routes


def get_tx_nodes(assignment, routing_tables):
    tx_nodes = set()
    for helpee, helper in enumerate(assignment):
        routing_path = vehicle.route.get_routing_path(helpee, helper, routing_tables)
        for cnt, node in enumerate(routing_path):
            if cnt != len(routing_path) - 1:
                tx_nodes.add(node)
    return tx_nodes


def get_neighbor_map(assignment, routing_tables):
    neighbor_map = {}
    for helpee, helper in enumerate(assignment):
        for node in vehicle.route.get_routing_path(helpee, helper, routing_tables):
            neighbor_map[node] = vehicle.route.get_neighbors(node, routing_tables)
    return neighbor_map

def get_valid_neighbor_map(neighbor_map, nodes_on_routes, tx_nodes, assignment, routing_tables):
    # valid neighbor: either a transmitting neighbor or a next hop neighbor
    valid_neighbor_map = {}
    for k, v in neighbor_map.items():
        if k in nodes_on_routes:
            valid_neighbors = []
            for i in v:
                if i in tx_nodes:
                    valid_neighbors.append(i)
            valid_neighbor_map[k] = valid_neighbors
    for helpee, helper in enumerate(assignment):
        routing_path = vehicle.route.get_routing_path(helpee, helper, routing_tables)
        for cnt, node in enumerate(routing_path):
            if cnt != len(routing_path) - 1:
                next_hop = routing_path[cnt + 1]
                if next_hop not in valid_neighbor_map[node]:
                    valid_neighbor_map[node].append(next_hop)
    return valid_neighbor_map


def random_sched(num_of_helpees, num_of_helpers, random_seed):
    random.seed(random_seed)
    assignments = find_all_one_to_one(num_of_helpees, num_of_helpers)
    return assignments[random.randint(0, len(assignments) - 1)]


def min_total_distance_sched(num_of_helpees, num_of_helpers, positions):
    print("Using the min total distance sched")
    distances = {}
    for assignment in find_all_one_to_one(num_of_helpees, num_of_helpers):
        distance = 0
        for cnt, node in enumerate(assignment):
            distance += get_distance(positions[cnt], positions[node])
        distances[get_id_from_assignment(assignment)] = distance
    sorted_distances = sorted(distances.items(), key=lambda item: item[1])
    return get_assignment_from_id(sorted_distances[0][0])


def coverage_aware_sched(num_of_helpees, num_of_helpers, positions, coverages):
    """
        Assign helpers to helpees considering the coverage of each helpee 
        Primary goal: Make as many helpers reachable by its corresponding helppee
        Secondry goal: Minimize the total distance

    Args:
        num_of_helpees (int): the number of helpees
        num_of_helpers (int): the number of helpers
        positions (list): the postion (x, y) of each node  
        coverages (list): the radio range of each node
    """
    in_range_nums = {}
    distances = {}
    for assignment in find_all_one_to_one(num_of_helpees, num_of_helpers):
        distance = 0
        in_range_num = 0
        for cnt, node in enumerate(assignment):
            tx_position = positions[cnt]
            rx_position = positions[node]
            assignment_id = get_id_from_assignment(assignment)
            distance += get_distance(tx_position, rx_position)
            if is_in_range(tx_position, rx_position, coverages[cnt]):
                in_range_num += 1
        distances[assignment_id] = distance
        in_range_nums[assignment_id] = in_range_num
        # print(assignment, in_range_nums[assignment_id], distance)
    increasing_distances = sorted(distances.items(), key=lambda item: item[1])
    max_in_range_num = 0
    max_assignment_id = increasing_distances[0][0]
    # print(increasing_distances)
    for item in increasing_distances:
        assignment_id = item[0]
        in_range_num = in_range_nums[assignment_id]
        # print(assignment_id, in_range_num, item[1])
        if in_range_num > max_in_range_num:
            max_in_range_num = in_range_num
            max_assignment_id = assignment_id
    return get_assignment_from_id(max_assignment_id)


def wwan_bw_sched(num_of_helpees, num_of_helpers, bws):
    print("Using the bw sched")
    print(bws)
    scores = {}
    for assignment in find_all_one_to_one(num_of_helpees, num_of_helpers):
        bw = 0
        for cnt, node in enumerate(assignment):
            bw += bws[node]
        scores[get_id_from_assignment(assignment)] = bw
    sorted_scores = sorted(scores.items(), key=lambda item: -item[1]) # decreasing order
    return get_assignment_from_id(sorted_scores[0][0])


def wwan_bw_distance_sched(num_of_helpees, num_of_helpers, bws, positions, p):
    print("Using the bw-distance sched")
    scores = {}
    for assignment in find_all_one_to_one(num_of_helpees, num_of_helpers):
        distance = 0
        bw = 0
        for cnt, node in enumerate(assignment):
            distance += get_distance(positions[cnt], positions[node])
            bw += bws[node]
        scores[get_id_from_assignment(assignment)] = bw - p * distance
    sorted_scores = sorted(scores.items(), key=lambda item: -item[1]) # decreasing order
    return get_assignment_from_id(sorted_scores[0][0])


def route_sched(num_of_helpees, num_of_helpers, routing_tables):
    print("Using the routeAware sched")
    scores = {}
    for assignment in find_all_one_to_one(num_of_helpees, num_of_helpers):
        print(assignment)
        num_hops = 0
        nodes_on_routes = get_nodes_on_routes(assignment, routing_tables)
        tx_nodes = get_tx_nodes(assignment, routing_tables)
        print(nodes_on_routes)
        print(tx_nodes)
        neighbor_map = get_neighbor_map(assignment, routing_tables)
        print(neighbor_map)
        valid_neighbor_map = get_valid_neighbor_map(neighbor_map, nodes_on_routes, tx_nodes, assignment, routing_tables)
        print(valid_neighbor_map)
        interference_count = 0
        for helpee, helper in enumerate(assignment):
            routing_path = vehicle.route.get_routing_path(helpee, helper, routing_tables)
            for cnt, node in enumerate(routing_path):
                # if cnt == 0:
                #     interference_count += count
                # else:
                count = len(valid_neighbor_map[node])
                    # for i in valid_neighbor_map[node]:
                    #     if routing_path[cnt - 1] == i:
                    #         count -= 1
                interference_count += count
        print(interference_count)
        scores[get_id_from_assignment(assignment)] = 0 - interference_count
    print(scores)
    sorted_scores = sorted(scores.items(), key=lambda item: -item[1]) # decreasing order
    return get_assignment_from_id(sorted_scores[0][0])


def bipartite():
    print("Solving bipartite matching")


def main():
    print("Schedule helpers for helpees")
    # find_all_one_to_one(3, 5)
    # min_total_distance_sched(1, 2, [[0, 0], [1, 1], [2, 2]])
    # assignment = coverage_aware_sched(2, 2, [[0, 0], [1, 1], [2, 2], [3, 3]], [2, 2, 2, 2])
    routing_tables = routing_tables = {4: {0: 2, 1: 2, 2: 2, 3: 3, 5: 5}, 2: {0: 0, 1: 1, 3: 3, 4: 4, 5: 5}, 
                      5: {0: 2, 1: 2, 2: 2, 3: 3, 4: 4}, 3: {0: 0, 1: 1, 2: 2, 4: 4, 5: 5}, 
                      1: {0: 0, 2: 2, 3: 3, 4: 2, 5: 2}, 0: {1: 1, 2: 2, 3: 3, 4: 2, 5: 2}}
    assignment = route_sched(2, 4, routing_tables)
    print("Solution:", assignment)


if __name__ == "__main__":
    main()
