#!/bin/bash
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=2:mem=10gb
#PBS -j oe



# Cluster Environment Setup
ncpus=2
hostname -i

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate RLSB3

# Copy over the whole dir


cd rl-training

python3 launch_parallel_sb3.py -n $ncpus 

exit 0
