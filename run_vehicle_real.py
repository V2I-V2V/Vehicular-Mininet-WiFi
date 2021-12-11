import os
import sys
import getpass
import time
import subprocess
import signal
from datetime import datetime

# sync time
os.system('sudo ntpdate NTP-server-dili')

start_time = time.time()

id_to_data_dir = ['86', '97', '108', '119', '130', '141', '152', '163', '174', '185']
helpee_confs = ['no-helpee', 'single-helpee-1', 'two-helpees']

dataset_dir = '/home/' + getpass.getuser() + '/Carla/lidar/'


vehicle_id = int(sys.argv[1])
helpee_number = int(sys.argv[2])
scheme = sys.argv[3]

adaptive_encode = 0
if 'adapt' or 'harbor' in scheme:
    adaptive_encode = 1
v2v_mode = 0
if 'v2v' in scheme:
    v2v_mode = 1

data_dir = dataset_dir + id_to_data_dir[vehicle_id] + '/'
helpee_conf = helpee_confs[helpee_number] + '.txt'
time_to_next_min = time.time() + 60 - time.time() % 60

# start the gps deamon
os.system('adb forward tcp:20175 tcp:50000')

if 'v2v' in scheme:
    # start v2v server
    server_cmd = "python3 -u server/server.py -s v2v -n 3 --v2v_mode 1 --data_type Carla -t ./input/traces/constant.txt > %d-server.log"%int(start_time)
    server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
    time.sleep(2)

cmd = 'python3 -u vehicle/vehicle.py -i %d --v2v_mode %d -t %f -d %s -l input/locations/0.txt --adaptive %d -c input/helpee_conf/%s > %d-node%d-%s.txt' % \
    (vehicle_id, v2v_mode, time_to_next_min, data_dir, adaptive_encode, helpee_conf, int(start_time), vehicle_id, helpee_confs[helpee_number])

# start the vehicle application
proc = subprocess.Popen(cmd,stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

while time.time() < time_to_next_min:
    time.sleep(0.005)

print("Start driving the vehicle now ...")
# sleep for 1 min
time.sleep(60)

# kill the process
os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

if 'v2v' in scheme:
    os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)

print("Experiment finished.")
