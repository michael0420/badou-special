#!/bin/bash
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -o /gs/hs1/tga-i/jorai/Exercise/Week_11/VGG/221230_2250_VGG_log

. /etc/profile.d/modules.sh
module load cuda
export PATH="/gs/hs1/tga-i/jorai/Environment/anaconda3/bin:${PATH}"
source activate pim
export MASTER_ADDR=$(ip addr show dev ib0 | grep '\<inet\>' | cut -d " " -f 6 | cut -d "/" -f 1)

python train.py