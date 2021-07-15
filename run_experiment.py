import getpass
import os, sys
import time
from datetime import datetime, timedelta
import subprocess


def parse_config_from_file(filename):
    config_params = {}
    config_file = open(filename, 'r')
    for line in config_file.readlines():
        if "=" in line:
            key = line.split("=")[0]
            value = line.split("=")[1][:-1]  # remove "\n"
            config_params[key] = value
    config_file.close()
    return config_params


def write_config_to_file(config_params, filename):
    config_file = open(filename, 'w')
    for k, v in config_params.items():
        config_file.write(k + "=" + v + "\n")
    config_file.close()


def init_config():
    input_path = os.path.dirname(os.path.abspath(__file__)) + "/input"
    config_params = {"num_of_nodes": "6", "location_file": input_path + "/locations/location-mindist-bw.txt",
                     "network_trace": input_path + "/traces/trace-mindist-bw.txt", "ptcl_config": input_path + "/pcds/pcd-data-config.txt",
                     "scheduler": "minDist", "fps": "10", "t": "70", "helpee_conf": input_path + "/helpee_conf/helpee-nodes.txt",
                     "routing": "custom", "frames": "300"}
    return config_params


def kill_mininet():
    cmd = "sudo mn -c"
    os.system(cmd)
    print("+" + cmd)


def clean_output():
    cmds = ["sudo rm " + os.path.dirname(os.path.abspath(__file__)) + "/output/*.bin", 
            "sudo rm " + os.path.dirname(os.path.abspath(__file__)) + "/pcaps/*.pcap",
            "sudo rm " + os.path.dirname(os.path.abspath(__file__)) + "/logs/*.log",
            "sudo rm " + os.path.dirname(os.path.abspath(__file__)) + "/logs/*.route"]
    for cmd in cmds:
        os.system(cmd)
        print("+" + cmd)


def create_folder():
    now = datetime.now() # current date and time
    folder = "data-" + now.strftime("%m%d%H%M")
    while os.path.exists(folder):
        print(folder, "exists")
        now += timedelta(minutes=1)
        folder = "data-" + now.strftime("%m%d%H%M")
    return folder


def run_experiment(config_params):
    cmd = "sudo python3 " +  os.path.dirname(os.path.abspath(__file__)) + "/vehicular_perception.py -n " +\
         config_params["num_of_nodes"] + " -l " + config_params["location_file"] + " --trace " + config_params["network_trace"] + " -p " + config_params["ptcl_config"] + " -s " + config_params["scheduler"] + " --helpee_conf " + config_params["helpee_conf"] +\
         " -t " + config_params["t"] + " --fps " + config_params["fps"] + " --run_app" + " -r " + config_params["routing"]
    os.system(cmd)
    print("+" + cmd)


def check_exception_in_output():
    logs = os.path.dirname(os.path.abspath(__file__)) + "/logs/"
    proc = subprocess.Popen("grep -nr \"Traceback\" %s"%logs, stdout=subprocess.PIPE, shell=True)
    (output, err) = proc.communicate()
    print("+checking output")
    if len(output) != 0:
        print('+Error found in logs')
        os.system('touch '+os.path.dirname(os.path.abspath(__file__)) + "/logs/error.log")
        # sys.exit(1)
    # if output 

def move_output(folder):
    cmds = ["cp -r " + os.path.dirname(os.path.abspath(__file__)) + "/logs/ " + folder, 
            "cp -r " + os.path.dirname(os.path.abspath(__file__)) + "/pcaps/ " + folder]
    for cmd in cmds:
        os.system(cmd)
        print("+" + cmd)


def run_analysis(folder, config_params):
    cmd = "sudo python3 " + os.path.dirname(os.path.abspath(__file__)) \
        + "/analysis-scripts/calc_delay.py " + folder + '/ ' +  config_params["num_of_nodes"] \
        + ' ' + config_params["frames"] + ' >/dev/null'
    os.system(cmd)


def main():
    for i in range(10):
        kill_mininet()
        clean_output()
        folder = create_folder()
        cmd = "mkdir " + folder
        os.system(cmd)
        print("+", cmd)
        config_params = init_config()
        print(config_params)
        write_config_to_file(config_params, folder + "/config.txt")
        # config_params = parse_config_from_file(folder + "/config.txt")
        run_experiment(config_params)
        check_exception_in_output()
        move_output(folder)
        run_analysis(folder, config_params)
    

if __name__ == "__main__":
    main()
