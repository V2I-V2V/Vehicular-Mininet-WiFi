import numpy as np
import sys

start_loc = np.loadtxt(sys.argv[1])[0]
loc = start_loc
end_loc = np.loadtxt(sys.argv[1])[1]
duration = int(sys.argv[2])

# speed_100ms = speed/10.0
shift = end_loc - start_loc

increase_step = shift/10/duration

print(increase_step)

for i in range(duration*10+1):
    loc = np.vstack((loc, start_loc+increase_step*i))
#     print(loc)

print(loc.shape)
np.savetxt(sys.argv[1].split('.')[0]+"-full.txt", loc, fmt='%f')
