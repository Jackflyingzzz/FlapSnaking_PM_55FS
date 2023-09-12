#!/bin/bash
#PBS -lwalltime=00:30:00
#PBS -lselect=1:ncpus=1:mem=8gb

cp $PBS_O_WORKDIR/incompact3d.prm $TMPDIR/
cp $PBS_O_WORKDIR/alpha_transitions.prm $TMPDIR/
cp $PBS_O_WORKDIR/probe_layout.txt $TMPDIR/

$PBS_O_WORKDIR/incompact3d
JOB_ID_SPLIT=${PBS_JOBID%.*}
mkdir $HOME/$JOB_ID_SPLIT
cp * $HOME/$JOB_ID_SPLIT
