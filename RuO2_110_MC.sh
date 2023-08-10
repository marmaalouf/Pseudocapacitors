#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l pmem=4gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -A ixd4_g_g_bc_default
#PBS -N MC_None

cd $PBS_O_WORKDIR

python RuO2_110_MC.py --rootdir Strain1_Files/ --name None 