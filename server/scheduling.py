# Schedule helpers for helpees according to different algorithms
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vehicle.route
import itertools
import math
import random
import statistics

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


def find_all(num_of_helpees, num_of_helpers):
    
    def get_assignments(n):
        if n == 1:
            return [(x + num_of_helpees,) for x in range(num_of_helpers)]
        assignments = []
        last_assignments = get_assignments(n - 1)
        for i in range(len(last_assignments)):
            for j in range(num_of_helpers):
                assignments.append(last_assignments[i] + (j + num_of_helpees,))
        return assignments

    return get_assignments(num_of_helpees)


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_counts(assignment):
    counts = {}
    for helper in assignment:
        if helper not in counts:
            counts[helper] = 1
        else:
            counts[helper] += 1
    return counts


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


def random_sched(num_of_helpees, num_of_helpers, random_seed, is_one_to_one=False):
    random.seed(random_seed)
    assignments = find_all_one_to_one(num_of_helpees, num_of_helpers) if is_one_to_one else find_all(num_of_helpees, num_of_helpers)
    return assignments[random.randint(0, len(assignments) - 1)]


def get_distances(assignment, positions):
    distances = []
    for helpee, helper in enumerate(assignment):
        distances.append(get_distance(positions[helpee], positions[helper]))
    return distances


def get_distance_scores(assignment, positions):
    # list, each element is the score for a path in the assignment
    scores = []
    for helpee, helper in enumerate(assignment):
        max_distance = 0
        for i in range(len(assignment), len(positions)):
            distance_to_helpee = get_distance(positions[helpee], positions[i])
            if distance_to_helpee > max_distance:
                max_distance = distance_to_helpee
        # print(max_distance)
        scores.append(1 - get_distance(positions[helpee], positions[helper]) / max_distance)
    # print(assignment, scores)
    return scores


def min_total_distance_sched(num_of_helpees, num_of_helpers, positions, is_one_to_one=False):
    print("Using the min total distance sched")
    sum_distances = {}
    assignments = find_all_one_to_one(num_of_helpees, num_of_helpers) if is_one_to_one else find_all(num_of_helpees, num_of_helpers)
    for assignment in assignments:
        sum_distance = 0
        for distance in get_distances(assignment, positions):
            sum_distance += distance
        sum_distances[get_id_from_assignment(assignment)] = sum_distance
    sorted_sum_distances = sorted(sum_distances.items(), key=lambda item: item[1])
    return get_assignment_from_id(sorted_sum_distances[0][0])


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


def get_v2i_bws(assignment, bws):
    v2i_bws = []
    for helpee, helper in enumerate(assignment):
        v2i_bws.append(bws[helper])
    return v2i_bws


def wwan_bw_sched(num_of_helpees, num_of_helpers, bws, is_one_to_one=False):
    print("Using the bw sched")
    scores = {}
    assignments = find_all_one_to_one(num_of_helpees, num_of_helpers) if is_one_to_one else find_all(num_of_helpees, num_of_helpers)
    for assignment in assignments:
        bw = 0
        counts = get_counts(assignment)
        for helpee, helper in enumerate(assignment):
            bw += bws[helper] / (counts[helper] + 1)
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


def get_path_interference_count(routing_path, valid_neighbor_map):
    interference_count = 0
    for node in routing_path:
        count = len(valid_neighbor_map[node])
        interference_count += count
    return interference_count


def get_interference_counts(assignment, routing_tables):
    nodes_on_routes = get_nodes_on_routes(assignment, routing_tables)
    tx_nodes = get_tx_nodes(assignment, routing_tables)
    neighbor_map = get_neighbor_map(assignment, routing_tables)
    valid_neighbor_map = get_valid_neighbor_map(neighbor_map, nodes_on_routes, tx_nodes, assignment, routing_tables)
    interference_counts = []
    for helpee, helper in enumerate(assignment):
        routing_path = vehicle.route.get_routing_path(helpee, helper, routing_tables)
        interference_counts.append(get_path_interference_count(routing_path, valid_neighbor_map))
    return interference_counts


def route_sched(num_of_helpees, num_of_helpers, routing_tables, is_one_to_one=False):
    print("Using the routeAware sched")
    scores = {}
    assignments = find_all_one_to_one(num_of_helpees, num_of_helpers) if is_one_to_one else find_all(num_of_helpees, num_of_helpers)
    for assignment in assignments:
        sum_interference_count = 0
        for interference_count in get_interference_counts(assignment, routing_tables):
            sum_interference_count += interference_count
        scores[get_id_from_assignment(assignment)] = 0 - sum_interference_count
    sorted_scores = sorted(scores.items(), key=lambda item: -item[1]) # decreasing order
    return get_assignment_from_id(sorted_scores[0][0])


def get_bw_scores(assignment, v2i_bws):
    scores = []
    for helpee, helper in enumerate(assignment):
        score = 1
        counts = get_counts(assignment)
        if 5 < v2i_bws[helpee] / (counts[helper] + 1) < 25:
            score = (v2i_bws[helpee] - 5) / 20
        elif v2i_bws[helpee] <= 5:
            score = 0
        scores.append(score)
    return scores


def get_interference_scores(assignment, interference_counts, routing_tables):
    # print(assignment)
    scores = []
    for helpee, helper in enumerate(assignment):
        interference_count = interference_counts[helpee]
        routing_path = vehicle.route.get_routing_path(helpee, helper, routing_tables)
        neighbor_map = get_neighbor_map(assignment, routing_tables)
        max_interference_count = get_path_interference_count(routing_path, neighbor_map)
        nodes_on_routes = get_nodes_on_routes(assignment, routing_tables)
        path_nodes = set(routing_path)
        min_neighbor_map = get_valid_neighbor_map(neighbor_map, nodes_on_routes, path_nodes, assignment, routing_tables)
        min_interference_count = get_path_interference_count(routing_path, min_neighbor_map)
        scores.append(1 - (interference_count - min_interference_count) / (max_interference_count - min_interference_count))
        # print(1 - (interference_count - min_interference_count) / (max_interference_count - min_interference_count), interference_count, max_interference_count, min_interference_count)
    return scores


def combined_sched(num_of_helpees, num_of_helpers, positions, bws, routing_tables, is_one_to_one=False):
    print("Using the combined sched")
    scores = {}
    assignments = find_all_one_to_one(num_of_helpees, num_of_helpers) if is_one_to_one else find_all(num_of_helpees, num_of_helpers)
    for assignment in assignments:
        # distances = get_distances(assignment, positions)
        v2i_bws = get_v2i_bws(assignment, bws)
        interference_counts = get_interference_counts(assignment, routing_tables)
        distance_scores = get_distance_scores(assignment, positions)
        bw_scores = get_bw_scores(assignment, v2i_bws)
        interference_scores = get_interference_scores(assignment, interference_counts, routing_tables)
        # print(interference_scores)
        scores[get_id_from_assignment(assignment)] = statistics.harmonic_mean(distance_scores) + statistics.harmonic_mean(bw_scores) + statistics.harmonic_mean(interference_scores)
        # print(assignment, scores[get_id_from_assignment(assignment)], 
        #       statistics.harmonic_mean(distance_scores), statistics.harmonic_mean(bw_scores), statistics.harmonic_mean(interference_scores))
    sorted_scores = sorted(scores.items(), key=lambda item: -item[1]) # decreasing order
    return get_assignment_from_id(sorted_scores[0][0])


def bipartite():
    print("Solving bipartite matching")


def main():
    print(find_all(3, 3))


if __name__ == "__main__":
    main()
