import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server.scheduling
import threading


def test_minDist():
    assignment = server.scheduling.min_total_distance_sched(2, 2, [[0, 0], [1, 1], [2, 2], [3, 3]], True)
    assert assignment == (3, 2), "Wrong assignment"
    assignment = server.scheduling.min_total_distance_sched(2, 2, [[0, 0], [1, 1], [2, 2], [3, 3]], False)
    assert assignment == (2, 2), "Wrong assignment"


def test_bwAware():
    assignment = server.scheduling.wwan_bw_sched(2, 3, [1, 2, 13, 4, 5], True)
    assert assignment == (2, 4), "Wrong assignment for bwAware"
    assignment = server.scheduling.wwan_bw_sched(2, 2, [1, 1, 1, 10], False)
    assert assignment == (3, 3), "Wrong assignment for bwAware"
    

def test_routeAware():
    routing_tables = {4: {0: 2, 1: 2, 2: 2, 3: 3, 5: 5}, 2: {0: 0, 1: 1, 3: 3, 4: 4, 5: 5}, 
                      5: {0: 2, 1: 2, 2: 2, 3: 3, 4: 4}, 3: {0: 0, 1: 1, 2: 2, 4: 4, 5: 5}, 
                      1: {0: 0, 2: 2, 3: 3, 4: 2, 5: 2}, 0: {1: 1, 2: 2, 3: 3, 4: 2, 5: 2}}
    assignment = server.scheduling.route_sched(2, 4, routing_tables)
    print("Solution:", assignment)


def test_combined():
    routing_tables = {4: {0: 2, 1: 2, 2: 2, 3: 3, 5: 5}, 2: {0: 0, 1: 1, 3: 3, 4: 4, 5: 5}, 
                      5: {0: 2, 1: 2, 2: 2, 3: 3, 4: 4}, 3: {0: 0, 1: 1, 2: 2, 4: 4, 5: 5}, 
                      1: {0: 0, 2: 2, 3: 3, 4: 2, 5: 2}, 0: {1: 1, 2: 2, 3: 3, 4: 2, 5: 2}}
    positions = [(0, 0), (40, 0), (0, 40), (40, 40), (0, 80), (40, 80)]
    bws = [10 for x in range(6)]
    assignment = server.scheduling.combined_sched(2, 4, positions, bws, routing_tables)[0]
    assert assignment == (2, 3), "Wrong assignment for combined"


def test_combined_specific():
    positions = [(0, 0), (50, 0), (50, 100), (50, 50), (0, 100), (0, 50), (0, 150), (50, 150)]
    routing_tables = {0: {1: 1, 2: 1, 3: 3, 4: 4, 5: 5, 6: 1, 7: 1}, 
                      1: {0: 0, 2: 2, 3: 3, 4: 0, 5: 5, 6: 2, 7: 2}, 
                      2: {0: 1, 1: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, 
                      3: {0: 0, 1: 1, 2: 2, 4: 4, 5: 5, 6: 2, 7: 7}, 
                      4: {0: 0, 1: 0, 2: 2, 3: 3, 5: 5, 6: 6, 7: 7}, 
                      5: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 6, 7: 2}, 
                      6: {0: 2, 1: 2, 2: 2, 3: 2, 4: 4, 5: 2, 7: 7}, 
                      7: {0: 2, 1: 2, 2: 2, 3: 3, 4: 4, 5: 2, 6: 6}}
    bws = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0] 
    assignment = server.scheduling.combined_sched(3, 5, positions, bws, routing_tables)[0]
    print("Choose assignment:", assignment)
    assert assignment == (4,4,7), "Wrong assignment for combined"


def test_random():
    assignment = server.scheduling.random_sched(2, 3, 100)
    print(assignment)


def test_coverage_aware():
    assignment = server.scheduling.coverage_aware_sched(2, 2, [[0, 0], [1, 1], [2, 2], [3, 3]], [2, 2, 2, 2])
    print(assignment)


def test_find_assignments():
    assignments = server.scheduling.find_all_one_to_one(3, 5)
    print(assignments)


def test_intf_cnt():
    pass

if __name__ == "__main__":
    # test_minDist()
    # test_bwAware()
    # test_routeAware()
    # test_combined()
    # test_random()
    # test_coverage_aware()
    # test_find_assignments()
    test_combined_specific()
