#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l pmem=4gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -N MC4_St0_pH|13
#PBS -A ixd4_g_g_bc_default

cd $PBS_O_WORKDIR

python RuO2_110_MC_4.py --rootdir Strain0_Files/ --name Strain0 --pH 13 