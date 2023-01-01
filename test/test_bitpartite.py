import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.match import *
import matplotlib.pyplot as plt
from pulp import *
import time
import numpy as np
from collections import defaultdict

# from_nodes = [1, 2, 3, 4]
# to_nodes = [1, 2, 3, 4, 5, 6]
# ucap = {1: 1, 2: 1, 3: 1, 4: 1} #u node capacities
# vcap = {1: 1, 2: 1, 3: 2, 4: 1, 5: 3, 6: 2} #v node capacities

# wts = {(1, 1): 0.5, (1, 3): 0.3, (2, 1): 0.4, (2, 4): 0.1, (3, 2): 0.7, (3, 4): 0.2, (4, 1): 0.5, (4, 3): 0.3, (4, 2): 0.4, (4, 4): 0.1, (3, 2): 0.7, (3, 4): 0.2,
#        (1, 5): 0.5, (1, 6): 0.3, (2, 5): 0.4, (2, 6): 0.1, (3, 5): 0.7, (3, 6): 0.2, (4, 5): 0.5, (4, 6): 0.3, (3, 4): 0.4, (3, 5): 0.1, (3, 6): 0.7, (3, 1): 0.2}


def solve_wbm(from_nodes, to_nodes, wt, ucap, vcap):
    ''' A wrapper function that uses pulp to formulate and solve a WBM'''

    prob = LpProblem("WBM_Problem", LpMaximize)

    # Create The Decision variables
    choices = LpVariable.dicts("e",(from_nodes, to_nodes), 0, 1, LpInteger)
    print(choices.keys())

    # Add the objective function 
    prob += lpSum([wt[u][v] * choices[u][v] 
                   for u in from_nodes
                   for v in to_nodes]), "Total weights of selected edges"


    # Constraint set ensuring that the total from/to each node 
    # is less than its capacity
    for u in from_nodes:
        for v in to_nodes:
            prob += lpSum([choices[u][v] for v in to_nodes]) <= ucap[u], ""
            prob += lpSum([choices[u][v] for u in from_nodes]) <= vcap[v], ""


    # The problem data is written to an .lp file
    # prob.writeLP("WBM.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    # print( "Status:", LpStatus[prob.status])
    return(prob)


def print_solution(prob):
    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        if v.varValue > 1e-3:
            print(f'{v.name} = {v.varValue}')
    print(f"Sum of wts of selected edges = {round(value(prob.objective), 4)}")


def get_selected_edges(prob):

    selected_from = [v.name.split("_")[1] for v in prob.variables() if v.value() > 1e-3]
    selected_to   = [v.name.split("_")[2] for v in prob.variables() if v.value() > 1e-3]

    selected_edges = []
    for su, sv in list(zip(selected_from, selected_to)):
        selected_edges.append((su, sv))
    return(selected_edges)        



# wt = create_wt_doubledict(from_nodes, to_nodes, wts)
# p = solve_wbm(from_nodes, to_nodes, wt)
# start_t = time.time()
# p = solve_wbm(from_nodes, to_nodes, wt)
# print("solve takes", time.time() - start_t)
# print_solution(p)


def test():
    np.random.seed(0)
    from_nodes = np.arange(1, 9, dtype=np.int).tolist()
    to_nodes = np.arange(1, 13, dtype=np.int).tolist()
    ucap = defaultdict(list)
    vcap = defaultdict(list)

    for fnode in from_nodes:
        ucap[fnode] = 1
    
    for tnode in to_nodes:
        vcap[tnode] = 8
    
    wts = {}
    
    for fnode in from_nodes:
        for tnode in to_nodes:
            wts[(fnode, tnode)] = (np.random.choice(7, 1)[0] + 1) / 10.
    
    wt = create_wt_doubledict(from_nodes, to_nodes, wts)
    print(from_nodes, to_nodes)
    start_t = time.time()
    p = solve_wbm(from_nodes, to_nodes, wt, ucap, vcap)
    print("solve takes", time.time() - start_t)
    print_solution(p)
    
    
test()
