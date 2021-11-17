import getpass
import os, sys
import time
from datetime import datetime, timedelta
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.analyze_single_exp import single_exp_analysis


def parse_config_setting(filename, sched=['v2v', 'v2i', 'v2v-adapt', 'v2i-adapt', 'combined-adapt']):
    settings = []
    config_file = open(filename, 'r')
    i = 0
    for line in config_file.readlines():
        parse = line.split()
        num_node = parse[0].split('=')[1]
        loc = parse[1].split('=')[1]
        bw = parse[2].split('=')[1]
        helpee_conf = parse[3].split('=')[1]
        for s in sched:
            setting = (num_node, s, loc, bw, helpee_conf)
            i += 1
            settings.append(setting)
    return settings


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
                     "routing": "custom", "frames": "300", "one_to_many": '1', "adaptive_encode": "0",
                     "adapt_frame_skipping": "0",
                     "add_loc_noise": "0",
                     "adapt_frame_skipping": "0", "combine_method": "op_sum", "score_method": "harmonic",
                     "v2v_mode": "0"}
    return config_params


def write_log():
    if os.path.exists('/home/'+ getpass.getuser() + '/exp_log.txt'):
        log = open('/home/'+ getpass.getuser() + '/exp_log.txt', 'a+')
    else:
        log = open('/home/'+ getpass.getuser() + '/exp_log.txt', 'w+')
    cwd_str = os.getcwd()
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    cwd_str = dt_string + ' ' + cwd_str + '\n'
    log.write(cwd_str)
    log.close()


def write_exp_config(setting_repeat):
    if os.path.exists('config_summary.txt'):
        summary = open('config_summary.txt', 'a+')
    else:
        summary = open('config_summary.txt', 'w+')
    for setting in setting_repeat:
        summary_str = "num_node="+ setting[4] +"\tloc="+setting[0] + "\tbw="+setting[1] + "\thelpee_config="+setting[2] + '\n' # \
                        # "\trepeat="+setting[3] + '\n'
        summary.write(summary_str)
        
    summary.close()


def kill_mininet(n):
    cmd = "sudo mn -c"
    for i in range(n):
        os.system(cmd)
        print("+" + cmd)


def kill_routing():
    cmd = "sudo pkill olsrd"
    os.system(cmd)
    print("+" + cmd)


def kill_application():
    cmd = "sudo kill $(ps aux | grep \"[v]ehicular_perception.py\" | awk {'print $2'})"
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


def run_experiment(config_params, is_save_data=False, is_run_app=True, is_tcpdump_enabled=False):
    cmd = "sudo python3 " +  os.path.dirname(os.path.abspath(__file__)) + "/vehicular_perception.py -n " +\
         config_params["num_of_nodes"] + " -l " + config_params["location_file"] + " --trace " +\
         config_params["network_trace"] + " -p " + config_params["ptcl_config"] + " -s " + config_params["scheduler"] +\
         " --helpee_conf " + config_params["helpee_conf"] +\
         " -t " + config_params["t"] + " --fps " + config_params["fps"] + " -r " + config_params["routing"] +\
         " --multi " + config_params["one_to_many"] + " --adaptive_encode " + config_params["adaptive_encode"] +\
         " --combine_method " + config_params["combine_method"] + " --score_method " + config_params["score_method"]
    if config_params["adapt_frame_skipping"] == "1":
        cmd += " --adapt_frame_skipping"
    if config_params["add_loc_noise"] == "1":
        cmd += " --add_noise_to_loc"
    if config_params["v2v_mode"] == "1":
        cmd += " --v2v_mode"
    if is_save_data:
        cmd += " --save_data 1"
    if is_tcpdump_enabled:
        cmd += " --collect-traffic"
    if is_run_app:
        cmd += " --run_app"

    os.system(cmd)
    print("+" + cmd)


def check_exception_in_output():
    logs = os.path.dirname(os.path.abspath(__file__)) + "/logs/"
    proc = subprocess.Popen("grep -nr \"Traceback\" %s"%logs, stdout=subprocess.PIPE, shell=True)
    (output, err) = proc.communicate()
    tb_output = output.decode('utf-8').split('\n')
    proc = subprocess.Popen("grep -nr \"ConnectionResetError:\" %s"%logs, stdout=subprocess.PIPE, shell=True)
    (cr_output, err) = proc.communicate() 
    cr_output = cr_output.decode('utf-8').split('\n')
    print("[INFO] checking output")
    if len(tb_output) != len(cr_output):
        print('[INFO] Error found in logs')
        os.system('echo \"' + output.decode('utf-8') + '\" > ' + os.path.dirname(os.path.abspath(__file__))\
                    + "/logs/error.log")
        # sys.exit(1)
    # if output 

def move_output(folder, is_save_data=False):
    cmds = ["cp -r " + os.path.dirname(os.path.abspath(__file__)) + "/logs/ " + folder, 
            "cp -r " + os.path.dirname(os.path.abspath(__file__)) + "/pcaps/ " + folder]
    if is_save_data:
        cmds.append("cp -r " + os.path.dirname(os.path.abspath(__file__)) + "/output/ " + folder)
    for cmd in cmds:
        os.system(cmd)
        print("+" + cmd)


def run_analysis(folder, config_params):
    single_exp_analysis(folder+'/', int(config_params["num_of_nodes"]), config_params["network_trace"], \
         config_params["location_file"], config_params["helpee_conf"], int(config_params["t"]), config_params)



def main():
    # print(parse_config_setting('/home/mininet-wifi/v2x_exp_comprehensive/config_summary.txt'))
    print("This is juat a template on running experiments. Please do not direcly call\
             python3 run_experiment.py. Exiting.....")
    sys.exit(1)
    scheds = ['bwAware', 'random', 'minDist']
    locs = [os.path.dirname(os.path.abspath(__file__)) + "/input/locations/" + x \
        for x in [ 'location-multihop.txt', '106.txt', '5.txt']]
    bw_traces = [os.path.dirname(os.path.abspath(__file__)) + "/input/traces/" + x \
         for x in ['lte-4.txt', 'lte-15.txt', 'lte-22.txt']]
    helpee_confs = [os.path.dirname(os.path.abspath(__file__)) + "/input/helpee_conf/" + x \
         for x in ['helpee-start.txt', 'helpee-start-middle.txt', 'helpee-middle-middle.txt']]
    settings = []
    for i in range(3):
        for sched in scheds:
            for loc in locs:
                for bw_trace in bw_traces:
                    for helpee_conf in helpee_confs:
                        settings.append((i, sched, loc, bw_trace, helpee_conf))
    # for cnt, setting in enumerate(settings): 
    #     print(cnt, setting)
    start = 31
    for setting in settings[start:]:
        i, sched, loc, bw_trace, helpee_conf = setting
        kill_mininet(3)
        kill_routing()
        kill_application()
        clean_output()
        folder = create_folder()
        cmd = "mkdir " + folder
        os.system(cmd)
        print("+", cmd)
        config_params = init_config()
        config_params["scheduler"] = sched
        config_params["location_file"] = loc
        config_params["network_trace"] = bw_trace
        config_params["helpee_conf"] = helpee_conf
        print(config_params)
        write_config_to_file(config_params, folder + "/config.txt")
        # config_params = parse_config_from_file(folder + "/config.txt")
        run_experiment(config_params)
        check_exception_in_output()
        move_output(folder)
        run_analysis(folder, config_params)
    

if __name__ == "__main__":
    main()
