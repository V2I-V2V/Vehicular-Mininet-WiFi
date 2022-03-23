# Schedule helpers for helpees according to different algorithms
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vehicle.route
import itertools
import math
import random
import statistics
import time

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
        # avoid the distance to be 0
        scores.append(1 - get_distance(positions[helpee], positions[helper]) / max_distance)
    # print(assignment, scores)
    return scores


def get_max_distances(num_helpees, positions):
    max_dist = {}
    for helpee in range(num_helpees):
        max_distance = 0
        for i in range(num_helpees, len(positions)):
            distance_to_helpee = get_distance(positions[helpee], positions[i])
            if distance_to_helpee > max_distance:
                max_distance = distance_to_helpee  
        max_dist[helpee] = max_distance
    return max_dist   


def min_total_distance_sched(num_of_helpees, num_of_helpers, positions, is_one_to_one=False):
    print("Using the min total distance sched")
    print(positions)
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
    # print("Using the bw sched")
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
    if len(routing_path) == 0:
        # if routing path empty (no valid path), return a large intererence cnt
        return 65535
    for node in routing_path:
        # print(routing_path, valid_neighbor_map[node])
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
        # print(helpee, helper)
        routing_path = vehicle.route.get_routing_path(helpee, helper, routing_tables)
        interference_counts.append(get_path_interference_count(routing_path, valid_neighbor_map))
    # print('intf count', interference_counts)
    return interference_counts


def route_sched(num_of_helpees, num_of_helpers, routing_tables, is_one_to_one=False):
    # print("Using the routeAware sched")
    scores = {}
    assignments = find_all_one_to_one(num_of_helpees, num_of_helpers) if is_one_to_one else find_all(num_of_helpees, num_of_helpers)
    for assignment in assignments:
        sum_interference_count = 0
        print(get_interference_counts(assignment, routing_tables))
        for interference_count in get_interference_counts(assignment, routing_tables):
            sum_interference_count += interference_count
        scores[get_id_from_assignment(assignment)] = 0 - sum_interference_count
        print("assignment ", assignment, 0 - sum_interference_count)
    sorted_scores = sorted(scores.items(), key=lambda item: -item[1]) # decreasing order
    return get_assignment_from_id(sorted_scores[0][0])


def get_bw_scores(assignment, v2i_bws):
    # print(assignment, v2i_bws)
    scores = []
    for helpee, helper in enumerate(assignment):
        score = 1
        counts = get_counts(assignment)
        average_bw = v2i_bws[helpee] / (counts[helper] + 1)
        # print(average_bw)
        if 0.64 < average_bw < 10:
            score = (average_bw - 0.64) / 10
        elif average_bw <= 0.64:
            score = 0.001
        scores.append(score)
    return scores


def get_interference_scores(assignment, interference_counts, routing_tables, positions):
    # print(assignment)
    scores = []
    not_reachable_cnt = 0 
    for helpee, helper in enumerate(assignment):
        interference_count = interference_counts[helpee]
        routing_path = vehicle.route.get_routing_path(helpee, helper, routing_tables)
        # print("routing path", routing_path)
        neighbor_map = get_neighbor_map(assignment, routing_tables)
        # print("neighbour map", neighbor_map)
        max_interference_count = get_path_interference_count(routing_path, neighbor_map)
        nodes_on_routes = get_nodes_on_routes(assignment, routing_tables)
        path_nodes = set(routing_path)
        min_neighbor_map = get_valid_neighbor_map(neighbor_map, nodes_on_routes, path_nodes, assignment, routing_tables)
        # print("min neighbour map", min_neighbor_map)
        min_interference_count = get_path_interference_count(routing_path, min_neighbor_map)
        # print('intf score components: ic(pi,A) %d, ic(pi,pi) %d, ic(pi,G) %d'%\
        #     (interference_count, min_interference_count, max_interference_count))
        # check if every node in path is in range
        reachable = True
        for node_idx in range(len(routing_path)-1):
            if not is_in_range(positions[routing_path[node_idx]], positions[routing_path[node_idx+1]], 130):
                reachable = False
                print("Non reachiable pair in path ", routing_path)
                break
        if len(routing_path) == 0 or interference_count > max_interference_count or not reachable:
            score = 0
            not_reachable_cnt += 1
        elif max_interference_count != min_interference_count:
            score = 1 - (interference_count - min_interference_count) / (max_interference_count - min_interference_count)        
        else:
            score = 1
        scores.append(score)
        # print(1 - (interference_count - min_interference_count) / (max_interference_count - min_interference_count), interference_count, max_interference_count, min_interference_count)
    return scores, not_reachable_cnt


def get_score(scores, score_method="harmonic"):
    if score_method == "harmonic":
        return statistics.harmonic_mean(scores)
    elif score_method == "min":
        return min(scores)
    else:
        print("score method does not exist")


def get_path_score(distance_score, bw_score, interference_score, score_method):
    if score_method == "sum":
        return distance_score + bw_score + interference_score
    elif score_method == "min":
        print("use min path score", distance_score, bw_score, interference_score)
        return min(distance_score, bw_score, interference_score)


def get_combined_scores(distance_scores, bw_scores, interference_scores, combine_method, score_method):
    if combine_method == "op_sum":
        return get_score(distance_scores, score_method) + get_score(bw_scores, score_method) + get_score(interference_scores, score_method)
    elif combine_method == "op_min":
        min_score = 3
        min_idx = 0
        for i in range(len(distance_scores)):
            path_score = get_path_score(distance_scores[i], bw_scores[i], interference_scores[i], score_method)
            print("path min score", path_score, min_score)
            min_score = min(min_score, path_score)
        return min_score
    else:
        print("combine method not supported yet")


def combined_sched(num_of_helpees, num_of_helpers, positions, bws, routing_tables, is_one_to_one=False, combine_method="op_sum", score_method="harmonic"):
    # print("Using the combined sched", num_of_helpees, num_of_helpers, positions, bws, routing_tables)
    # print("Assignment dist_score bw_score intf_score")
    # print(routing_tables)
    scores, scores_dist, scores_bw, scores_intf, assignment_reachable_cnt  = {}, {}, {}, {}, {}
    scores_combined_base, scores_dist_min, scores_bw_min, scores_intf_min = {}, {}, {}, {}
    assignments = find_all_one_to_one(num_of_helpees, num_of_helpers) if is_one_to_one else find_all(num_of_helpees, num_of_helpers)
    # max_dist_dict = get_max_distances(num_of_helpees, positions)
    for assignment in assignments:
        # distances = get_distances(assignment, positions)
        v2i_bws = get_v2i_bws(assignment, bws)
        interference_counts = get_interference_counts(assignment, routing_tables)
        # print("Intf cnt ", interference_counts)
        distance_scores = get_distance_scores(assignment, positions)
        bw_scores = get_bw_scores(assignment, v2i_bws)
        interference_scores, not_reachable_cnt = get_interference_scores(assignment, interference_counts, routing_tables, positions)
        # print("scores: ", distance_scores, bw_scores, interference_scores)
        # print("assignment score:", assignment, "%.3f"%statistics.harmonic_mean(distance_scores), "%.3f"%statistics.harmonic_mean(bw_scores), "%.3f"%statistics.harmonic_mean(interference_scores))
        assignment_id = get_id_from_assignment(assignment)
        scores_dist[assignment_id] = get_score(distance_scores, score_method)
        scores_bw[assignment_id] = get_score(bw_scores, score_method)
        scores_intf[assignment_id] = get_score(interference_scores, score_method)
        assignment_reachable_cnt[assignment_id] = num_of_helpees - not_reachable_cnt
        scores[assignment_id] = get_combined_scores(distance_scores, bw_scores, interference_scores, combine_method, score_method)
        if not_reachable_cnt > 0:
            scores[get_id_from_assignment(assignment)] = 0
        # print(assignment, scores[get_id_from_assignment(assignment)], 
        #       statistics.harmonic_mean(distance_scores), statistics.harmonic_mean(bw_scores), statistics.harmonic_mean(interference_scores))

    sorted_scores = sorted(scores.items(), key=lambda item: -item[1]) # decreasing order
    max_in_range_num = 0
    selected_score = scores[sorted_scores[0][0]]
    max_assignment_id = sorted_scores[0][0]
    for item in sorted_scores:
        assignment_id = item[0]
        in_range_num = assignment_reachable_cnt[assignment_id]
        # print(assignment_id, in_range_num, item[1])
        if in_range_num > max_in_range_num:
            max_in_range_num = in_range_num
            max_assignment_id = assignment_id
            selected_score = scores[item[0]]

    sorted_base_scores = sorted(scores_combined_base.items(), key=lambda item: -item[1]) # decreasing order
    # print("Selected score:", selected_score, get_assignment_from_id(max_assignment_id))
    # selected_score = scores[sorted_scores[0][0]]
    # print("Scores harmonic: ", \
    #     scores_dist_min[sorted_base_scores[0][0]], scores_bw_min[sorted_base_scores[0][0]], scores_intf_min[sorted_base_scores[0][0]],\
    #     scores_dist_min[sorted_scores[0][0]], scores_bw_min[sorted_scores[0][0]], scores_intf_min[sorted_scores[0][0]], \
    #     time.time())
    # print("Best choice (min_sum/combined) scores: ", sorted_scores[0][0], sorted_base_scores[0][0], \
    #     scores[sorted_scores[0][0]], scores[sorted_base_scores[0][0]], \
    #     scores_combined_base[sorted_scores[0][0]], scores_combined_base[sorted_base_scores[0][0]])


    # print("Best scores:", max(scores_dist, key=scores_dist.get), max(scores_bw,  key=scores_bw.get), max(scores_intf, key=scores_intf.get))
    # print("Worst scores: ", min(scores_dist, key=scores_dist.get), min(scores_bw,  key=scores_bw.get), min(scores_intf, key=scores_intf.get))
    return get_assignment_from_id(max_assignment_id), selected_score, scores


def bipartite():
    print("Solving bipartite matching")


def main():
    print(find_all(3, 3))


if __name__ == "__main__":
    main()
