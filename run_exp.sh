for t in 1 2 3 4
do
    echo $t
    # run an experiment
    sudo python3 vehicular_perception.py -n 6 -p input/pcds/pcd-data-config.txt -l input/locations/location-example-full.txt --trace input/traces/trace-mindist-bw.txt --helpee_conf ./input/helpee_conf/helpee-nodes.txt --run_app -s minDist -t 90 --fps 10
    # copy output data to another directory
    tag=$(date +"%m%d%H%M%S")
    mkdir ~/v2x
    mkdir ~/v2x/data-$tag
    cp -r logs ~/v2x/data-$tag/
    cp -r output ~/v2x/data-$tag/
    cp -r pcaps ~/v2x/data-$tag/
    python3 analysis-scripts/calc_delay.py ~/v2x/data-$tag/ 6 240
done