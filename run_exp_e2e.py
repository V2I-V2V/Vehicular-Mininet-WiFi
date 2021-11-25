from run_experiment import *


def main():
    scheds = ['v2v', 'v2i', 'v2v-adapt', 'v2i-adapt'] # 'v2v', 'v2i', 'v2v-adapt', 'v2i-adapt', 'combined-adapt'
    settings = parse_config_setting('/home/mininet-wifi/v2x_exp_comprehensive/config_town05.txt', sched=scheds)
    write_log()
    start = 0
    for setting in settings[start:]:
        num_nodes, sched, loc, bw_trace, helpee_conf = setting
        loc = os.path.dirname(os.path.abspath(__file__)) + "/input/locations/" + loc
        bw_trace = os.path.dirname(os.path.abspath(__file__)) + "/input/traces/" + bw_trace
        helpee_conf = os.path.dirname(os.path.abspath(__file__)) + "/input/helpee_conf/" + helpee_conf
        kill_mininet(1)
        kill_routing()
        kill_application()
        clean_output()
        folder = create_folder()
        cmd = "mkdir " + folder
        os.system(cmd)
        print("+", cmd)
        config_params = init_config()
        config_params["ptcl_config"] = loc.replace("locations", "pcds")
        config_params["num_of_nodes"] = num_nodes
        if sched == 'combined-adapt':
            config_params["scheduler"] = 'combined'
            config_params['adaptive_encode'] = '1'
        else:
            config_params["scheduler"] = sched
        if 'adapt' in sched:
            config_params['adaptive_encode'] = '1'
            config_params["scheduler"] = sched[:-6]
        if 'v2v' in sched:
            config_params["v2v_mode"] = "1"
        config_params["location_file"] = loc
        config_params["network_trace"] = bw_trace
        config_params["helpee_conf"] = helpee_conf
        # config_params["add_loc_noise"] = "1"
        config_params["fps"] = "10"
        config_params["routing"] = 'olsrd'
        if sched == 'combined-adapt':
            config_params["routing"] = 'custom'
        config_params["t"] = '50'
        # input_path = os.path.dirname(os.path.abspath(__file__)) + "/input"
        # config_params["ptcl_config"] = input_path + "/pcds/carla-data-config.txt"
        print(config_params)
        write_config_to_file(config_params, folder + "/config.txt")
        # config_params = parse_config_from_file(folder + "/config.txt")
        run_experiment(config_params, is_save_data=False, is_run_app=True, is_tcpdump_enabled=False)
        check_exception_in_output()
        move_output(folder, is_save_data=False)
        run_analysis(folder, config_params)
    

if __name__ == "__main__":
    main()


# if __name__ == '__main__':
#     print(parse_config_setting('/home/mininet-wifi/v2x_exp_comprehensive/config_summary.txt'), ['v2v', 'v2i', 'v2v-adapt', 'v2i-adapt'])