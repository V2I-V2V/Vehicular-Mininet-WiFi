import os
import sys
import getpass
import time
import subprocess
import signal

start_time = time.time()

id_to_data_dir = ['86', '97', '108', '119', '130', '141', '152', '163', '174', '185']
helpee_confs = ['no-helpee', 'single-helpee-1', 'two-helpees']

dataset_dir = '/home/' + getpass.getuser() + '/Carla/lidar/'


vehicle_id = int(sys.argv[1])
helpee_number = int(sys.argv[2])

data_dir = dataset_dir + id_to_data_dir[vehicle_id] + '/'
helpee_conf = helpee_confs[helpee_number] + '.txt'
time_to_next_min = time.time() + 60 - time.time() % 60

# start the gps deamon
os.system('adb forward tcp:20175 tcp:50000')

cmd = 'python3 vehicle/vehicle.py -i %d -t %f -d %s -l input/locations/0.txt --adaptive 1 -c input/helpee_conf/%s > %d-node%d-%s.txt' % \
    (vehicle_id, time_to_next_min, data_dir, helpee_conf, int(start_time), vehicle_id, helpee_confs[helpee_number])

# start the vehicle application
proc = subprocess.Popen(cmd,stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

# sleep for 1.33 min
time.sleep(80)

# kill the process
os.killpg(os.getpgid(proc.pid), signal.SIGTERM)


