for t in 1
do
    echo $t
    # run an experiment
    sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-mindist-bw.txt --trace input/traces/trace-mindist-bw.txt --helpee_conf ./input/helpee_conf/helpee-nodes.txt --run_app -s minDist -t 70 --fps 10
    # copy output data to another directory
    tag='minDist-'$(date +"%m%d%H%M%S")
    mkdir ~/v2x
    mkdir ~/v2x/data-$tag
    cp -r logs ~/v2x/data-$tag/
    cp -r output ~/v2x/data-$tag/
    cp -r pcaps ~/v2x/data-$tag/
    python3 analysis-scripts/calc_delay.py ~/v2x/data-$tag/ 6 160
    sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-mindist-bw.txt --trace input/traces/trace-mindist-bw.txt --helpee_conf ./input/helpee_conf/helpee-nodes.txt --run_app -s bwAware -t 70 --fps 10
    # copy output data to another directory
    tag='bwAware-'$(date +"%m%d%H%M%S")
    mkdir ~/v2x
    mkdir ~/v2x/data-$tag
    cp -r logs ~/v2x/data-$tag/
    cp -r output ~/v2x/data-$tag/
    cp -r pcaps ~/v2x/data-$tag/
    python3 analysis-scripts/calc_delay.py ~/v2x/data-$tag/ 6 160
    sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-mindist-bw.txt --trace input/traces/trace-mindist-bw.txt --helpee_conf ./input/helpee_conf/helpee-nodes.txt --run_app -s combined -t 70 --fps 10
    # copy output data to another directory
    tag='combined-'$(date +"%m%d%H%M%S")
    mkdir ~/v2x
    mkdir ~/v2x/data-$tag
    cp -r logs ~/v2x/data-$tag/
    cp -r output ~/v2x/data-$tag/
    cp -r pcaps ~/v2x/data-$tag/
    python3 analysis-scripts/calc_delay.py ~/v2x/data-$tag/ 6 160
    sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-mindist-bw.txt --trace input/traces/trace-mindist-bw.txt --helpee_conf ./input/helpee_conf/helpee-nodes.txt --run_app -s random -t 70 --fps 10
    # copy output data to another directory
    tag='random-'$(date +"%m%d%H%M%S")
    mkdir ~/v2x
    mkdir ~/v2x/data-$tag
    cp -r logs ~/v2x/data-$tag/
    cp -r output ~/v2x/data-$tag/
    cp -r pcaps ~/v2x/data-$tag/
    python3 analysis-scripts/calc_delay.py ~/v2x/data-$tag/ 6 160
    sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-mindist-bw.txt --trace input/traces/trace-mindist-bw.txt --helpee_conf ./input/helpee_conf/helpee-nodes.txt --run_app -s routeAware -t 70 --fps 10
    # copy output data to another directory
    tag='routeAware-'$(date +"%m%d%H%M%S")
    mkdir ~/v2x
    mkdir ~/v2x/data-$tag
    cp -r logs ~/v2x/data-$tag/
    cp -r output ~/v2x/data-$tag/
    cp -r pcaps ~/v2x/data-$tag/
    python3 analysis-scripts/calc_delay.py ~/v2x/data-$tag/ 6 160
    sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-mindist-bw.txt --trace input/traces/trace-mindist-bw.txt --helpee_conf ./input/helpee_conf/helpee-nodes.txt --run_app -s fixed ./input/assignments/assignments-sample.txt 6 -t 70 --fps 10
    # copy output data to another directory
    tag='fixed-'$(date +"%m%d%H%M%S")
    mkdir ~/v2x
    mkdir ~/v2x/data-$tag
    cp -r logs ~/v2x/data-$tag/
    cp -r output ~/v2x/data-$tag/
    cp -r pcaps ~/v2x/data-$tag/
    python3 analysis-scripts/calc_delay.py ~/v2x/data-$tag/ 6 160
done