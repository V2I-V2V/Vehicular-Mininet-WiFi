import networkx as nx
from pulp import *


def create_wt_doubledict(from_nodes, to_nodes, wts):

    wt = {}
    for u in from_nodes:
        wt[u] = {}
        for v in to_nodes:
            wt[u][v] = 0

    for k,val in wts.items():
        u,v = k[0], k[1]
        wt[u][v] = val
    return(wt)