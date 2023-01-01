import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.scheduling import combined_sched, random_sched, combined_sched_new
import time
import random



def test_computation_time(num_of_helpees, num_of_helpers, positions, bws, routing_tables):
    start = time.time()
    combined_sched_new(num_of_helpees, num_of_helpers, positions, bws, routing_tables)
    end = time.time()
    random_start = time.time()
    # random_sched(num_of_helpees, num_of_helpers, 42)
    combined_sched(num_of_helpees, num_of_helpers, positions, bws, routing_tables)
    random_end = time.time()
    return end-start, random_end-random_start 


def generate_routing(num_of_helpees, num_of_helpers):
    routing_tables = {}
    for i in range(num_of_helpees+num_of_helpers):
        route = {}
        for j in range(num_of_helpees+num_of_helpers):
            if j != i:
                route[j] = j
        routing_tables[i] = route
    return routing_tables

def generate_positions(num_of_helpees, num_of_helpers):
    positions = []
    for i in range(num_of_helpees+num_of_helpers):
        x, y = random.randrange(0, 140), random.randrange(0, 140)
        positions.append((x, y))
    return positions

def generate_bws(num_of_helpees, num_of_helpers):
    bw = []
    for i in range(num_of_helpees+num_of_helpers):
        bw.append(random.randrange(0, 40))
    return bw

if __name__ == '__main__':
    random.seed(10)
    file = open('overhead-new.txt', 'w')
    for helpee_num in range(8, 9):
        for helper_num in range(12, 13):
            positions, bw = generate_positions(helpee_num, helper_num), generate_bws(helpee_num, helper_num)
            routing_tables = generate_routing(helpee_num, helper_num)
            combined_time, random_time = test_computation_time(helpee_num, helper_num, positions, bw, routing_tables)
            file.write(str(helpee_num) + ' ' + str(helper_num) + ' ' + str(combined_time) +
             ' ' + str(random_time) + '\n')
            print(helpee_num, helper_num)
