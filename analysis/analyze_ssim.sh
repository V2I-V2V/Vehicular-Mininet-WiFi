# $1: prefix of folders
# $2: number of frames (-1 means all)
# $3: reference frame folder
dir=$PWD
echo $dir
for folder in *; do
    # echo $folder
    if [[ $folder == $1* ]]; then
        echo $folder
        # python3 -W ignore ~/Vehicular-Mininet-WiFi/ptcl/pcd_process.py dis $folder
        cd ~/Vehicular-Mininet-WiFi/analysis/ssim
        python2 compute_ssim.py $3 $dir/$folder/ $2 10
    fi
done
