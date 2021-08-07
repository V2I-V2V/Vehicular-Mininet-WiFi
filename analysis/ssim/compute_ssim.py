#!/usr/bin/python
import matlab.engine
import time
import os, sys


if __name__ == "__main__":
    # dir = "/home/shawnzhu/v2x/data-08011700/output/"
    ref_dir = sys.argv[1]
    dis_dir = sys.argv[2]
    nframes = int(sys.argv[3])
    if nframes == -1:
        nframes = 10000
    # cmd = "python3 ~/Vehicular-Mininet-WiFi/convert.py " + dir
    # os.system(cmd)
    start = time.time()
    eng = matlab.engine.start_matlab()
    now = time.time()
    print("time taken to start matlab: " + int(now - start) + "sec")
    [pcd_ssims, pcd_ids] = eng.compute_ssim(ref_dir, dis_dir, 1, nframes, nargout=2)
    print(len(pcd_ssims))
    ssims, ids = pcd_ssims[0], pcd_ids[0]
    for i in range(len(ssims)):
        print("%f %f" % (ids[i], ssims[i]))
    now = time.time()
    print("time taken in total: " + str(now - start) + " sec")
