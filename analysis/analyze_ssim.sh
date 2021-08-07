# $1: prefix of folders
# $2: number of frames (-1 means all)
dir=$PWD
echo $dir
for folder in *; do
    # echo $folder
    if [[ $folder == $1* ]]; then
        echo $folder
        cd /home/shawnzhu/Vehicular-Mininet-WiFi/analysis/ssim
        python compute_ssim.py /home/shawnzhu/gta_ref_frames/ $dir/$folder/output/ $2
    fi
done
