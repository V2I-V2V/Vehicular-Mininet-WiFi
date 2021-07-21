import sys, os
import numpy as np

log = sys.argv[1]
throughput = []

with open(log, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('[relay throughput]'):
            thrpt = float(line.split()[2])
            if thrpt != 0:
                throughput.append(thrpt)

print(throughput)
print(np.mean(throughput))
print(np.std(throughput))
