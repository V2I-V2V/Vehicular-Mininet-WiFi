import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server.scheduling

if __name__ == "__main__":
    print("test sched")
    assignment = server.scheduling.random_sched(2, 3, 100)
    assignment = server.scheduling.wwan_bw_sched(2, 3, [1, 2, 13, 4, 5])
    print(assignment)
    # server.scheduling.min_total_distance_sched()
