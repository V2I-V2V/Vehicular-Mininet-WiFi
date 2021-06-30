# Schedule helpers for helpees according to different algorithms

import itertools
import math

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


def wwan_bw_distance_sched(num_of_helpees, num_of_helpers, bws, positions):
    print("Using the bw-distance sched")
    scores = {}
    coverage = 60
    for assignment in find_all_one_to_one(num_of_helpees, num_of_helpers):
        distance = 0
        for cnt, node in enumerate(assignment):
            distance += get_distance(positions[cnt], positions[node])
        scores[get_id_from_assignment(assignment)] = distance / coverage 
    sorted_distances = sorted(scores.items(), key=lambda item: item[1])
    return get_assignment_from_id(sorted_distances[0][0])


def bipartite():
    print("Solving bipartite matching")


def main():
    print("Schedule helpers for helpees")
    # find_all_one_to_one(3, 5)
    # min_total_distance_sched(1, 2, [[0, 0], [1, 1], [2, 2]])
    assignment = coverage_aware_sched(2, 2, [[0, 0], [1, 1], [2, 2], [3, 3]], [2, 2, 2, 2])
    print("Solution:", assignment)


if __name__ == "__main__":
    main()
