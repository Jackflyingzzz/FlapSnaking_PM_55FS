#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=2:mem=20gb
#PBS -j oe



# Cluster Environment Setup


cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate RLSB3

# Copy over the whole dir


cd rl-training

python3 eval_sb3.py   

exit 0
