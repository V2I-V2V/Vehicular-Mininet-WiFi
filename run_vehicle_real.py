import os
import sys
import getpass
import time
import subprocess
import signal
from datetime import datetime



start_time = time.time()

now = datetime.now() # current date and time
time_str = now.strftime("%m%d%H%M")

id_to_data_dir = ['86', '97', '108', '119', '130', '141', '152', '163', '174', '185']
helpee_confs = ['no-helpee', 'single-helpee-1', 'two-helpees']

dataset_dir = '/home/' + getpass.getuser() + '/Carla/lidar/'


vehicle_id = int(sys.argv[1])
helpee_number = int(sys.argv[2])
scheme = sys.argv[3]

# sync time
os.system('sudo ntpdate NTP-server-dili')

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
    if vehicle_id == 0:
        server_cmd = "python3 -u server/server.py -s v2v -n 2 --v2v_mode 1 --data_type Carla -t ./input/traces/constant.txt > %s-server.log"%time_str
        server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        print("server started", time.time())
    #     time.sleep(1)
    else:
        time.sleep(3)

cmd = 'python3 -u vehicle/vehicle.py -i %d --v2v_mode %d -t %f -d %s -l input/locations/0.txt --adaptive %d -c input/helpee_conf/%s > %s-node%d-%s.txt' % \
    (vehicle_id, v2v_mode, time_to_next_min, data_dir, adaptive_encode, helpee_conf, time_str, vehicle_id, helpee_confs[helpee_number])

# start the vehicle application
proc = subprocess.Popen(cmd,stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

while time.time() < time_to_next_min:
    time.sleep(0.005)

time.sleep(30)
print("Start driving the vehicle now ...", time.time())
# sleep for 1 min
time.sleep(60)

# kill the process
os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

if 'v2v' in scheme:
    if vehicle_id == 0:
        os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        os.system('scp %s-server.log ryanzhu@dili.eecs.umich.edu:/z/ryanzhu/Vehicular-Mininet-WiFi/real_exp_logs/data-%s/logs/server.log'%(time_str, time_str))

os.system('scp %s-node%d-%s.txt ryanzhu@dili.eecs.umich.edu:/z/ryanzhu/Vehicular-Mininet-WiFi/real_exp_logs/data-%s/logs/node%d.log'%(time_str, vehicle_id, helpee_confs[helpee_number], time_str, vehicle_id))

print("Experiment finished.")
