import sys
import numpy as np
if len(sys.argv) < 4:
    print("usage python3 scale_bw.py <network_trace_file> <scale_bw_threshold> <scale_factor>")
    exit(0)
tracefile = sys.argv[1]
bw_threshold = int(sys.argv[2])
scale_factor = int(sys.argv[3])
trace = np.loadtxt(tracefile, dtype=float)
mask = trace > bw_threshold
trace[mask] /= scale_factor
np.savetxt(tracefile.split('.')[-2] + '-scaled.txt', trace, fmt='%f')
