#!/bin/bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
printf "\nqyes\n\nyes\n" | bash Anaconda3-2020.11-Linux-x86_64.sh # You could use other shells to
./anaconda3/bin/conda init bash
