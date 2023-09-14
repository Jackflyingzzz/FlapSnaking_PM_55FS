#!/bin/bash
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=72:mem=100gb:ngpus=1:gpu_type=RTX6000
#PBS -j oe



# Cluster Environment Setup
ncpus=72
hostname -i

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate fenicsproject

# Copy over the whole dir


cd rl-training

python3 launch_parallel_sb3.py -n $ncpus 

exit 0
